#include <mpi.h>
#include <nccl.h>
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/stream_executor/stream.h"
#include "gpu_types.h"

using namespace tensorflow;
using perftools::gputools::StreamExecutor;
using perftools::gputools::Stream;
using perftools::gputools::cuda::ScopedActivateExecutorContext;

#define MPI_CHECK(op, name) \
    { \
        int err_code = (op); \
        if (err_code) { \
            int err_len; \
            char err_buffer[MPI_MAX_ERROR_STRING]; \
            MPI_Error_string(err_code, err_buffer, &err_len); \
            std::cout << "operation " << name << " in " << __FILE__ << ":" << __LINE__ << " failed with err_code=" << \
                err_code << ": " << err_buffer << std::endl << std::flush; \
            MPI_Abort(MPI_COMM_WORLD, err_code); \
        } \
    }

#define NCCL_CHECK(op, name) \
    { \
        ncclResult_t err_code = (op); \
        if (err_code != ncclSuccess) { \
            std::cout << "operation " << name << " in " << __FILE__ << ":" << __LINE__ << " failed: \"" << \
                ncclGetErrorString(err_code) << "\"" << std::endl << std::flush; \
            exit(EXIT_FAILURE); \
        } \
    }

typedef std::function<void(Status)> DoneCallback;

class NcclComm
{
private:
    typedef struct CommEntry
    {
        void*          buffer_in;
        void*          buffer_out;
        size_t         count;
        ncclDataType_t dtype;
        int            op_num;
        CUevent        op_event;
        DoneCallback   done_callback;
        Status         status;
    } CommEntry;

    int mpi_size, mpi_rank, local_rank, global_rank, prereduce, global_size;
    ncclComm_t comm_local, comm_global;
    CUstream   nccl_stream;
    StreamExecutor* executor;
    //std::unique_ptr<Stream> stream;
    std::unique_ptr<Thread> thread_comm;
    std::unique_ptr<Thread> thread_done;
    mutex              mu_comm, mu_done;
    condition_variable cv_comm, cv_done;
    std::deque<CommEntry*> pending_comm GUARDED_BY(mu_comm);
    std::deque<CommEntry*> pending_done GUARDED_BY(mu_done);
    bool shutdown_requested = false;

public:
    NcclComm(int mpi_size, int mpi_rank, int local_rank, int global_rank, ncclUniqueId& local_id, ncclUniqueId& global_id, int prereduce, StreamExecutor* executor) :
        mpi_size(mpi_size),
        mpi_rank(mpi_rank),
        local_rank(local_rank),
        global_rank(global_rank),
        prereduce(prereduce),
        global_size(0),
        executor(executor)
    {
        if (mpi_rank != -1)
        {
            if (prereduce)
            {
                global_size = mpi_size / prereduce;
                NCCL_CHECK(ncclCommInitRank(&comm_local, prereduce, local_id, local_rank), "ncclCommInitRank Local");
                if (local_rank == 0 && global_size > 1)
                    NCCL_CHECK(ncclCommInitRank(&comm_global, global_size, global_id, global_rank), "ncclCommInitRank Global");
            }
            else
            {
                NCCL_CHECK(ncclCommInitRank(&comm_global, mpi_size, global_id, mpi_rank), "ncclCommInitRank");
            }
        }
        //printf("ncclCommInitRank: %d\n", mpi_rank);

        cuStreamCreate(&nccl_stream, CU_STREAM_DEFAULT);

        thread_comm.reset(Env::Default()->StartThread(ThreadOptions(), "nccl_comm_thread", [this] { NcclCommThread(); }));
        thread_done.reset(Env::Default()->StartThread(ThreadOptions(), "nccl_done_thread", [this] { NcclDoneThread(); }));
    }
    ~NcclComm()
    {
        {
            mutex_lock l1(mu_comm);
            mutex_lock l2(mu_done);
            shutdown_requested = true;
            cv_comm.notify_one();
            cv_done.notify_one();
        }
        if (mpi_rank != -1)
        {
            if (prereduce)
            {
                ncclCommDestroy(comm_local);
                if (local_rank == 0 && global_size > 1)
                    ncclCommDestroy(comm_global);
            }
            else
                ncclCommDestroy(comm_global);
        }
        cuStreamDestroy(nccl_stream);
    }
    void AddEntry(int op_num, const Tensor* data_in, const Tensor* data_out, DoneCallback done_callback, CUevent op_event)
    {
        // printf("AddEntry: %03d %d %s\n", op_num, mpi_rank, input->shape().DebugString().c_str());

        // Enque the entry for execution on the thread.
        mutex_lock l(mu_comm);
        pending_comm.push_front(new CommEntry({
            (void*)data_in->tensor_data().data(),
            (void*)data_out->tensor_data().data(),
            (size_t)data_in->NumElements(),
            data_in->dtype() == DT_HALF ? ncclHalf : ncclFloat,
            op_num, op_event, std::move(done_callback), Status::OK()
        }));
        cv_comm.notify_one();
    }
    void NcclCommThread()
    {
        // Activate appropriate context for this thread.
        ScopedActivateExecutorContext scoped_context(executor);

        while (true)
        {
            CommEntry* entry = NULL;
            {
                mutex_lock l(mu_comm);
                while (pending_comm.empty() || shutdown_requested)
                {
                    if (shutdown_requested)
                        return;
                    cv_comm.wait(l);
                }
                entry = pending_comm.back();
                pending_comm.pop_back();
            }
            // printf("NcclCommThread: %03d %d %d\n", entry->op_num, mpi_rank, (int)entry->input->NumElements());

            // Ensure the nccl op doesn't start executing before it would on the main compute stream.
            // This is executed on device and does not block.
            cuStreamWaitEvent(nccl_stream, entry->op_event, 0);

            ncclResult_t nccl_result = ncclSuccess;
            if (mpi_rank != -1)
            {
                if (prereduce)
                {
                    // reduce among each machine's gpu
                    nccl_result = ncclReduce(entry->buffer_in, entry->buffer_out, entry->count, entry->dtype, ncclSum, 0, comm_local, nccl_stream);
                    if (nccl_result == ncclSuccess)
                    {
                        // reduce among all machines with the root gpu on each
                        if (local_rank == 0 && global_size > 1)
                            nccl_result = ncclAllReduce(entry->buffer_in, entry->buffer_out, entry->count, entry->dtype, ncclSum, comm_global, nccl_stream);
                        // broadcast root gpu results out to the rest of the gpus on each machine
                        if (nccl_result == ncclSuccess)
                            nccl_result = ncclBroadcast(entry->buffer_in, entry->buffer_out, entry->count, entry->dtype, 0, comm_local, nccl_stream);
                    }
                }
                else
                {
                    // Simple all-reduce among all mpi ranks
                    nccl_result = ncclAllReduce(entry->buffer_in, entry->buffer_out, entry->count, entry->dtype, ncclSum, comm_global, nccl_stream);
                }
            }

            if (nccl_result != ncclSuccess)
            {
                entry->status = errors::Internal("Error invoking ncclAllReduce (", mpi_rank, "/", mpi_size, "): ", ncclGetErrorString(nccl_result));
                shutdown_requested = true;
            }

            // Wait for kernels to finish on another thread so this thread can continue to queue up work
            cuEventRecord(entry->op_event, nccl_stream);
            mutex_lock l(mu_done);
            pending_done.push_front(entry);
            cv_done.notify_one();
        }
    }
    void NcclDoneThread()
    {
        // Activate appropriate context for this thread.
        ScopedActivateExecutorContext scoped_context(executor);

        while (true)
        {
            CommEntry* entry = NULL;
            {
                mutex_lock l(mu_done);
                while (pending_done.empty())
                {
                    if (shutdown_requested)
                        return;
                    cv_done.wait(l);
                }
                entry = pending_done.back();
                pending_done.pop_back();
            }
            // printf("NcclDoneThread: %d\n", mpi_rank);

            cuEventSynchronize(entry->op_event); // too slow?
            // while (true)
            // {
            //     if (cuEventQuery(entry->op_event) == CUDA_ERROR_NOT_READY)
            //         // Polling is pretty cheap. Don't delay further execution after this reduce op.
            //         Env::Default()->SleepForMicroseconds(80);
            //     else
            //         break;
            // }
            entry->done_callback(entry->status);
            delete entry;
        }
    }
};

class NcclManager;
static mutex init_mu(LINKER_INITIALIZED);
static NcclManager* nccl_mgr_instance = NULL;

class NcclManager
{
private:
    int mpi_rank, mpi_size;
    MPI_Comm mpi_comm;

    std::vector<NcclComm*> comms, custom_comms;

    std::map<int, int> op_to_comm;
    std::map<std::pair<std::set<int>, int>, int> ranks_to_comm;
public:
    static NcclManager* instance(int num_comms, int prereduce, StreamExecutor* executor)
    {
        {
            mutex_lock l(init_mu);
            if (nccl_mgr_instance == NULL)
            {
                nccl_mgr_instance = new NcclManager(num_comms, prereduce, executor);

                atexit( [] { delete nccl_mgr_instance; } );
            }
        }
        return nccl_mgr_instance;
    }
    NcclManager(int num_comms, int prereduce, StreamExecutor* executor)
    {
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank), "MPI_Comm_rank");
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size), "MPI_Comm_size");

        int flag;
        MPI_CHECK(MPI_Initialized(&flag), "MPI_Initialized");
        if (!flag) {
            int required = MPI_THREAD_MULTIPLE;
            int provided = MPI_THREAD_SINGLE;
            MPI_CHECK(MPI_Init_thread(nullptr, nullptr, required, &provided), "MPI_Init_thread")
            if (required != provided) {
                std::cout << "MPI_Init_thread only provided level " << provided << " of multi-threading support while level "
                     << required << " is needed" << std::endl << std::flush;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm), "MPI_Comm_dup");

        comms.reserve(num_comms);
        while (comms.size() < num_comms)
        {
            int num_ids = 1;
            if (prereduce)
            {
                num_ids += mpi_size / prereduce;
                if ((mpi_size % prereduce) != 0)
                {
                    std::cout << "mpi_size not a multiple of prereduce size" << std::endl << std::flush;
                    exit(EXIT_FAILURE);
                }
            }
            std::vector<ncclUniqueId> ids;
            ids.reserve(num_ids);

            ncclUniqueId id;
            if (mpi_rank == 0)
                for (int i = 0; i < num_ids; i++)
                {
                    NCCL_CHECK(ncclGetUniqueId(&ids[i]), "ncclGetUniqueId");
                }
            MPI_CHECK(MPI_Bcast((void*)ids.data(), sizeof(ncclUniqueId) * num_ids, MPI_BYTE, 0, mpi_comm), "MPI_Bcast");

            int local_rank = -1, global_rank = -1;
            if (prereduce)
            {
                local_rank  = mpi_rank % prereduce;
                global_rank = mpi_rank / prereduce;
            }
            comms.push_back(new NcclComm(mpi_size, mpi_rank, local_rank, global_rank, ids[global_rank+1], ids[0], prereduce, executor));
        }
    }
    ~NcclManager()
    {
        for (auto comm : comms)
            delete comm;

        for (auto comm : custom_comms)
            delete comm;
    }

    void CreateCustomComm(int op_num, const std::set<int>& mpi_ranks, int comm_id, StreamExecutor* executor)
    {
        auto ranks_and_id = std::make_pair(mpi_ranks, comm_id);

        if (op_to_comm.count(op_num))
        {
            return;
        }
        else if (ranks_to_comm.count(ranks_and_id))
        {
            op_to_comm[op_num] = ranks_to_comm[ranks_and_id];
            return;
        }

        ncclUniqueId id;

        if (mpi_rank == 0)
        {
            NCCL_CHECK(ncclGetUniqueId(&id), "ncclGetUniqueId");
        }
        MPI_CHECK(MPI_Bcast((void*)(&id), sizeof(ncclUniqueId), MPI_BYTE, 0, mpi_comm), "MPI_Bcast");

        int custom_mpi_size = static_cast<int>(mpi_ranks.size());
        int custom_mpi_rank = -1;

        if (mpi_ranks.count(mpi_rank))
        {
            custom_mpi_rank = 0;

            for (auto r : mpi_ranks)
            {
                if (mpi_rank == r)
                {
                    break;
                }
                custom_mpi_rank++;
            }
        }

        if (custom_mpi_rank != -1)
        {
            custom_comms.push_back(new NcclComm(custom_mpi_size, custom_mpi_rank, -1, -1, id, id, 0, executor));
        }
        else
        {
            custom_comms.push_back(new NcclComm(-1, -1, -1, -1, id, id, 0, executor));
        }

        int custom_comm_idx = static_cast<int>(custom_comms.size()) - 1;
        ranks_to_comm[ranks_and_id] = custom_comm_idx;
        op_to_comm[op_num] = custom_comm_idx;
    }

    void AddEntry(int op_num, int comm_idx, const Tensor* data_in, const Tensor* data_out, DoneCallback done_callback, CUevent op_event)
    {
        if (op_to_comm.count(op_num))  // this op has a custom comm
        {
            auto custom_comm = custom_comms[op_to_comm[op_num]];
            custom_comm->AddEntry(op_num, data_in, data_out, std::move(done_callback), op_event);
        }
        else
        {
            comms[comm_idx]->AddEntry(op_num, data_in, data_out, std::move(done_callback), op_event);
        }
    }
};


REGISTER_OP("AllreduceNccl")
    .Input("local: T")
    .Output("global: T")
    .Attr("T: {float, half}")
    .Attr("op_num: int")
    .Attr("sync_size: int = 0")
    .Attr("num_comms: int = 2") // Needs to be constant across all ops
    .Attr("prereduce: int = 0") // Needs to be constant across all ops
    .Attr("logfile: string = ''")
    .Attr("mpi_rank: int = 0")
    .Attr("mpi_ranks: list(int) = []")
    .Attr("comm_id: int = 0")
    .Attr("debug_str: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
        ctx->set_output(0, ctx->input(0));
        return Status::OK();
    });

class AllreduceNcclOp : public AsyncOpKernel {
private:
    int op_num, num_comms, comm_idx, sync_size, prereduce, mpi_rank;
    CUevent op_event;
    std::string logfile;
    std::string debug_str;

    std::set<int> mpi_ranks;
    int comm_id;

public:
    explicit AllreduceNcclOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx), op_event(NULL)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("op_num",    &op_num   )); // This is the comm
        OP_REQUIRES_OK(ctx, ctx->GetAttr("mpi_rank",  &mpi_rank ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("sync_size", &sync_size));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_comms", &num_comms));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("prereduce", &prereduce));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("logfile",   &logfile  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("debug_str", &debug_str));

        std::vector<int> v_mpi_ranks;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("mpi_ranks", &v_mpi_ranks));
        for (auto r : v_mpi_ranks)
        {
            mpi_ranks.insert(r);
        }
        OP_REQUIRES_OK(ctx, ctx->GetAttr("comm_id", &comm_id));

        // Simple distribution of nccl ops to commuicator objects
        comm_idx = op_num % num_comms;
    }
    // Kernels executing on GPU tie very few resources on the CPU where the scheduler runs
    // We do not need a TF pool thread to perform the op
    bool IsExpensive() override { return false; }

    void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override
    {
        if (op_event == NULL)
            cuEventCreate(&op_event, CU_EVENT_DISABLE_TIMING ); // | CU_EVENT_BLOCKING_SYNC

        const Tensor& input = ctx->input(0);

        char debug_message[1000];
        bool verbose = !debug_str.empty();
        if (verbose)
        {
            sprintf(debug_message, " op_num: %03d size: %s debug_str: %s\n", op_num, input.shape().DebugString().c_str(), debug_str.c_str());
            std::cout << "START";
            std::cout << debug_message;
            fflush(stdout);
        }

        // TODO: Turn on for all ranks
        if (!logfile.empty())
            if (FILE* log = fopen(logfile.c_str(), "a"))
            {
              fprintf(log, "%03d %10d %12s %d %d %d\n",
                op_num,
                (int)input.NumElements(),
                input.shape().DebugString().c_str(),
                (int)ctx->step_id(),
                (int)ctx->frame_iter().frame_id,
                (int)ctx->frame_iter().iter_id
              );
              fclose(log);
            }

        // Nccl ops work in place, so save some memory here.
        // ctx->set_output(0, input);
        Tensor* output;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

        auto actual_done = [ctx, done, debug_message, verbose](Status s) {
            if (verbose)
            {
                std::cout << "END  ";
                std::cout << debug_message;
                fflush(stdout);
            }
            OP_REQUIRES_OK_ASYNC(ctx, s, done);
            done();
        };

        auto stream    = ctx->op_device_context()->stream();
        auto executor  = stream->parent();
        //auto gpu_info  = ctx->device()->tensorflow_gpu_device_info();
        //auto event_mgr = gpu_info->event_mgr;
        //int  gpu_id    = gpu_info->gpu_id;

        CUstream cu_stream = reinterpret_cast<CUstream>(stream->implementation()->GpuStreamHack());

        // Record an event to mark this ops insertion in the compute stream.
        // The reduce op will be executed on the nccl stream but
        // not before this event is reached on the compute stream.
        cuEventRecord(op_event, cu_stream);

        // this keeps the backward pass from gobbling up buffers for every reduce op.
        // By syncing we let big ones finish first.
        // It's ok to greedily execute small buffers.
        if (sync_size && input.NumElements() > sync_size)
            cuStreamSynchronize(cu_stream);

        auto manager = NcclManager::instance(num_comms, prereduce, executor);

        if (!mpi_ranks.empty())
        {
            manager->CreateCustomComm(op_num, mpi_ranks, comm_id, executor);
        }
        manager->AddEntry(op_num, comm_idx, &input, output, std::move(actual_done), op_event);
    }
    virtual ~AllreduceNcclOp()
    {
        cuEventDestroy(op_event);
    }
};
REGISTER_KERNEL_BUILDER(Name("AllreduceNccl").Device(DEVICE_GPU), AllreduceNcclOp);




REGISTER_OP("IdentitySynchronize")
    .Input( "x: n_out * T")
    .Output("y: n_out * T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("sync: bool = false")
    .Attr("sync_bwd: bool = true")
    .Attr("n_out: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* ctx)
    {
        int n_out; TF_RETURN_IF_ERROR(ctx->GetAttr("n_out", &n_out));
        for (int i = 0; i < n_out; i++)
            ctx->set_output(i, ctx->input(i));
        return Status::OK();
    })
    .Doc(R"doc(
Sync op to facilite nccl all-reduce calls.
Don't let the graph scheduler get too far ahead while waiting for asynch ops to complete.
)doc");

class IdentitySynchronizeOp : public OpKernel
{
public:
    explicit IdentitySynchronizeOp(OpKernelConstruction* ctx) : OpKernel(ctx)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("sync",  &sync ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("n_out", &n_out));
    }
    void Compute(OpKernelContext* ctx) override
    {
        if (sync)
        {
            CUstream stream = CTX_STREAM(ctx);
            cuStreamSynchronize(stream);
        }
        for (int i = 0; i < n_out; i++)
            ctx->set_output(i, ctx->input(i));
    }
    bool sync;
    int n_out;
};
REGISTER_KERNEL_BUILDER(Name("IdentitySynchronize").Device(DEVICE_GPU), IdentitySynchronizeOp);


// "Sylvain Jeaugey" <sjeaugey@nvidia.com>

