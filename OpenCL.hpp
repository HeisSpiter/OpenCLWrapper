///
/// \file    OpenCL.hpp
/// \brief   Simple wrapper class around OpenCL
/// \details This wrapper intends to provide simple methods to quickly get a
///          program working with OpenCL. Beware, it requires C++11.
/// \author  Pierre Schweitzer
/// \date    18-05-2012
///

#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <algorithm>

///
/// \namespace OpenCLWrapper
/// \brief     Main namespace
///
namespace OpenCLWrapper {

///
/// \enum    OpenCLParameters
/// \brief   Enumeration for all the supported OpenCL parameters
/// \details Those parameters can be set using SetParameter()
///
enum OpenCLParameters {
    TargetDevice,  ///< Select the prefered target device (CPU, GPU, ...)
    BuildOptions,  ///< Define parameters used during OpenCL kernel build
    MaxParameters  ///< Parameter index cannot be higher
};

///
/// \class OpenCL
/// \brief Main wrapper class
///
class OpenCL {
private:
    /// Event used to measure performances on any operation done by OpenCL
    cl::Event                 mEvent;
    /// Device index in mDevices that points to the used device
    unsigned int              mDevice;
    /// Context of the OpenCL execution
    cl::Context *             mContext;
    /// Options that will be used for kernel builds. Can be set with BuildOptions option
    std::string               mBuildOptions;
    /// Queue that will be used for OpenCL operations
    cl::CommandQueue *        mQueue;
    /// Type of the target device. Can be set with TargetDevice option
    cl_device_type            mTargetDevice;
    /// List of devices that are in the current context
    std::vector<cl::Device> * mDevices;

    ///
    /// \fn      OpenCL
    /// \param   Ocl The OpenCL instance to copy
    /// \brief   Copy constructor
    /// \details Disallow the copy constructor
    ///
    OpenCL(const OpenCL & Ocl) {
        // Do nothing
        (void)Ocl;
    }

    ///
    /// \fn      operator=
    /// \param   Ocl The OpenCL instance to affect to the other
    /// \return  The affected OpenCL instance
    /// \brief   Affectation operator
    /// \details Disallow the affectation operator
    ///
    OpenCL & operator=(const OpenCL & Ocl) {
        // Do nothing
        (void)Ocl;
        return *this;
    }

///
/// \def   INIT
/// \brief Generic macro used for initializing appropriate member
///
#define INIT(type)                         \
    if (m##type == 0) {                    \
        cl_int Error = Initialize##type(); \
        if (Error != CL_SUCCESS) {         \
            return Error;                  \
        }                                  \
    }

    ///
    /// \fn      InitializeDevices
    /// \return  CL_SUCCESS, CL_OUT_OF_HOST_MEMORY, CL_DEVICE_NOT_FOUND
    /// \brief   This function is used to find a device on which execute OpenCL
    /// \details It will browse all the availables platforms and all the available
    ///          devices to find the most suitable. Its behaviour can be changed
    ///          with the SetParameter() function and the TargetDevice parameter.
    ///
    cl_int InitializeDevices() {
#define BROWSE_DEVICES(type)                                               \
    for (unsigned int i = 0; i < Platforms.size(); i++) {                  \
        Platforms[i].getDevices(CL_DEVICE_TYPE_##type, mDevices);          \
        for (unsigned int j = 0; j < mDevices->size(); j++) {              \
            if (mDevices->at(j).getInfo<CL_DEVICE_AVAILABLE>() &&          \
                mDevices->at(j).getInfo<CL_DEVICE_COMPILER_AVAILABLE>()) { \
                mDevice = j;                                               \
                return CL_SUCCESS;                                         \
            }                                                              \
        }                                                                  \
    }

        std::vector<cl::Platform> Platforms;

        mDevices = new (std::nothrow) std::vector<cl::Device>;
        if (mDevices == 0) {
            return CL_OUT_OF_HOST_MEMORY;
        }
        cl::Platform::get(&Platforms);

        // First, browse for accelerator
        if (mTargetDevice == CL_DEVICE_TYPE_ALL || mTargetDevice &
            (CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_ACCELERATOR)) {
            BROWSE_DEVICES(ACCELERATOR)
        }

        // Then, GPU
        if (mTargetDevice == CL_DEVICE_TYPE_ALL || mTargetDevice &
            (CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_GPU)) {
            BROWSE_DEVICES(GPU)
        }

        // Finally, CPU
        if (mTargetDevice == CL_DEVICE_TYPE_ALL || mTargetDevice &
            (CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_CPU)) {
            BROWSE_DEVICES(CPU)
        }

        // In case no device was found ensure we reset mDevices
        delete mDevices;
        mDevices = 0;

        return CL_DEVICE_NOT_FOUND;
    }

    ///
    /// \fn      InitializeContext
    /// \return  Any of the OpenCL cl::Context error code and CL_OUT_OF_HOST_MEMORY
    /// \brief   This function is used to initialize the OpenCL context
    /// \details It will first look for devices if required.
    ///
    cl_int InitializeContext() {
        INIT(Devices);

        assert(mDevices != 0);

        cl_int Error;
        mContext = new (std::nothrow) cl::Context(*mDevices, 0, 0, 0, &Error);
        if (mContext == 0) {
            return CL_OUT_OF_HOST_MEMORY;
        }

        // In case of error ensure we reset mContext
        if (Error != CL_SUCCESS) {
            delete mContext;
            mContext = 0;
        }

        return Error;
    }

    ///
    /// \fn      InitializeQueue
    /// \return  Any of the OpenCL cl::CommandQueue error code and CL_OUT_OF_HOST_MEMORY
    /// \brief   This function is used to initialize the OpenCL queue
    /// \details It will first look for devices and initialize context if required.
    ///
    cl_int InitializeQueue() {
        INIT(Context);

        assert(mDevices != 0);
        assert(mContext != 0);

        cl_int Error;
        mQueue = new (std::nothrow) cl::CommandQueue(*mContext, mDevices->at(mDevice),
                                                     CL_QUEUE_PROFILING_ENABLE,
                                                     &Error);
        if (mQueue == 0) {
            return CL_OUT_OF_HOST_MEMORY;
        }

        // In case of error ensure we reset mQueue
        if (Error != CL_SUCCESS) {
            delete mQueue;
            mQueue = 0;
        }

        return Error;
    }

    ///
    /// \fn      ExecuteKernelFromKernelEx
    /// \param   Kernel   The kernel to execute
    /// \param   DataSize Size of data on which the kernel will work
    /// \param   Position Unused
    /// \return  Any of the OpenCL code for cl::Queue::enqueueNDRangeKernel
    /// \brief   This function queues any kernel for its execution on the target device
    /// \details According to the given DataSize it will compute an appropriate
    ///          grid size and queue the work item. An event is used for profiling.
    ///
    cl_int ExecuteKernelFromKernelEx(cl::Kernel & Kernel, long DataSize,
                                     long Position) {
        INIT(Queue);

        assert(mDevices != 0);
        assert(mContext != 0);
        assert(mQueue != 0);

        (void)Position;

        cl::NDRange GlobalSize, LocalSize;
        GetGridSize(LocalSize, GlobalSize, DataSize);

        return mQueue->enqueueNDRangeKernel(Kernel, cl::NullRange, GlobalSize,
                                            LocalSize, 0, &mEvent);
    }

    ///
    /// \fn      ExecuteKernelFromKernelEx
    /// \tparam  Arg        Type of the next kernel argument to queue
    /// \tparam  Args       Types of the last kernel arguments to queue
    /// \param   Kernel     The kernel to execute
    /// \param   DataSize   Size of data on which the kernel will work
    /// \param   Position   Position of the next argument for the kernel arguments
    /// \param   KernelArg  Next kernel argument to queue
    /// \param   KernelArgs Last kernels arguments to queue
    /// \return  Any of the OpenCL code for cl::Queue::enqueueNDRangeKernel
    /// \brief   This function queues any kernel for its execution on the target device
    /// \details According to the given DataSize it will compute an appropriate
    ///          grid size and queue the work item. An event is used for profiling.
    ///          But first, it will queue all the provided kernel arguments.
    ///
    template<typename Arg, typename... Args>
    cl_int ExecuteKernelFromKernelEx(cl::Kernel & Kernel, long DataSize,
                                     long Position, const Arg& KernelArg,
                                     const Args&... KernelArgs) {
        Kernel.setArg(Position, KernelArg);
        return ExecuteKernelFromKernelEx(Kernel, DataSize, Position + 1, KernelArgs...);
    }

public:
    ///
    /// \fn      OpenCL
    /// \brief   Constructor that simply initialized dummy context
    /// \details By default, no build option will be set and the class will look
    ///          for any suitable device to run on.
    ///
    OpenCL() {
        mTargetDevice = CL_DEVICE_TYPE_ALL;
        mBuildOptions = "";
        mContext = 0;
        mDevices = 0;
        mDevice = 0;
        mQueue = 0;
    }

    ///
    /// \fn    ~OpenCL
    /// \brief Destructor that simply release everything related to context
    ///
    ~OpenCL() {
        delete mContext;
        delete mDevices;
        delete mQueue;
    }

    ///
    /// \fn     AllocateBuffer
    /// \tparam T      Type of the elements in the buffer
    /// \param  Size   Number of elements in the buffer
    /// \param  Buffer Output buffer that will be allocated
    /// \return Any of the cl::Buffer error code
    /// \brief  Allocates a buffer on the target device
    ///
    template<typename T>
    cl_int AllocateBuffer(size_t Size, cl::Buffer & Buffer) {
        INIT(Context);

        assert(mDevices != 0);
        assert(mContext != 0);

        cl_int Error;
        Buffer = cl::Buffer(*mContext, CL_MEM_READ_WRITE, sizeof(T) * Size, 0, &Error);
        return Error;
    }

    ///
    /// \fn      GetGridSize
    /// \param   LocalSize  Number of work-items per work-group
    /// \param   GlobalSize Number of work-items
    /// \param   Size       Total size of the data to process
    /// \brief   Defines a computation size
    /// \warning This is only adapted for CUDA devices for the moment
    ///
    void GetGridSize(cl::NDRange & LocalSize, cl::NDRange & GlobalSize, long Size) const {
       const long MaxThreads = 512;

       if (Size <= MaxThreads)
       {
           LocalSize = cl::NDRange(std::min(MaxThreads, Size));
       }
       else
       {
           for (int i = MaxThreads; i > 0; i--) {
               if (!(Size % i)) {
                  LocalSize = cl::NDRange(i, Size / i);
                  break;
               }
           }
       }

       GlobalSize = cl::NDRange(Size);
    }

    ///
    /// \fn      GetUsedDevice
    /// \param   UsedDevice Device that will be used
    /// \return  CL_SUCCESS, CL_OUT_OF_HOST_MEMORY, CL_DEVICE_NOT_FOUND
    /// \brief   Return the device that will be used by OpenCL for compuation
    /// \details This function will search the computation device if not found
    ///          yet and will simply return it to the caller. This allow querying
    ///          its properties, for instance.
    ///
    cl_int GetUsedDevice(cl::Device & UsedDevice) {
        INIT(Devices);

        assert(mDevices != 0);

        UsedDevice = mDevices->at(mDevice);

        return CL_SUCCESS;
    }

    ///
    /// \fn      GetProgramFromFile
    /// \param   FileName File containing the source to build
    /// \param   Program  The built source code
    /// \return  Any of the OpenCL error of cl::Program and cl::Program::build
    /// \brief   This function builds the provided source code into a program
    /// \details This function will first read the file and then will use any
    ///          build option that may have been provided with the SetParameter()
    ///          with option BuildOptions. It will initialize a context first if
    ///          required.
    ///
    cl_int GetProgramFromFile(const char * FileName, cl::Program & Program) {
        std::ifstream Source(FileName);
        return GetProgramFromFile(Source, Program);
    }

    ///
    /// \fn      GetProgramFromFile
    /// \param   File    Kernel source code to build
    /// \param   Program The built source code
    /// \return  Any of the OpenCL error of cl::Program and cl::Program::build
    /// \brief   This function builds the provided source code into a program
    /// \details This function will first read the file and then will use any
    ///          build option that may have been provided with the SetParameter()
    ///          with option BuildOptions. It will initialize a context first if
    ///          required.
    ///
    cl_int GetProgramFromFile(std::ifstream & File, cl::Program & Program) {
        std::string Source(std::istreambuf_iterator<char>(File),
                           (std::istreambuf_iterator<char>()));
        return GetProgramFromSource(Source, Program);
    }

    ///
    /// \fn      GetProgramFromSource
    /// \param   Source  Kernel source code to build
    /// \param   Program The built source code
    /// \return  Any of the OpenCL error of cl::Program and cl::Program::build
    /// \brief   This function builds the provided source code into a program
    /// \details This function will use any build option that may have been provided
    ///          with the SetParameter() with option BuildOptions. It will
    ///          initialize a context first if required.
    ///
    cl_int GetProgramFromSource(const std::string & Source,
                                cl::Program & Program) {
        return GetProgramFromSource(Source.c_str(), Source.length(),
                                       Program);
    }

    ///
    /// \fn      GetProgramFromSource
    /// \param   Source  Kernel source code to build
    /// \param   Length  Length of the kernel source code
    /// \param   Program The built source code
    /// \return  Any of the OpenCL error of cl::Program and cl::Program::build
    /// \brief   This function builds the provided source code into a program
    /// \details This function will use any build option that may have been provided
    ///          with the SetParameter() with option BuildOptions. It will
    ///          initialize a context first if required.
    ///
    cl_int GetProgramFromSource(const char * Source, size_t Length,
                                cl::Program & Program) {
        INIT(Context);

        assert(mDevices != 0);
        assert(mContext != 0);

        cl_int Error;
        cl::Program::Sources Sources(1, std::make_pair(Source, Length + 1));
        Program = cl::Program(*mContext, Sources, &Error);
        if (Error !=  CL_SUCCESS) {
            return Error;
        }

        Error = Program.build(*mDevices, (mBuildOptions.empty() ? 0 : mBuildOptions.c_str()));
        return Error;
    }

    ///
    /// \fn      GetKernelFromFile
    /// \param   FileName   File containing the source with the kernel
    /// \param   KernelName The name of the kernel
    /// \param   Kernel     The matching kernel
    /// \return  Any of the OpenCL error of cl::Kernel, cl::Program and cl::Program::build
    /// \brief   This function returns a matching kernel from a source code
    /// \details This function will first read the provided file for source code,
    ///          and will then build the provided source code and will
    ///          finally return the matching kernel.
    /// \see     GetProgramFromFile()
    ///
    cl_int GetKernelFromFile(const char * FileName, const char * KernelName,
                             cl::Kernel & Kernel) {
        cl::Program Program;
        cl_int Error = GetProgramFromFile(FileName, Program);
        if (Error !=  CL_SUCCESS) {
            return Error;
        }

        return GetKernelFromProgram(Program, KernelName, Kernel);
    }

    ///
    /// \fn      GetKernelFromFile
    /// \param   File       File containing the source with the kernel
    /// \param   KernelName The name of the kernel
    /// \param   Kernel     The matching kernel
    /// \return  Any of the OpenCL error of cl::Kernel, cl::Program and cl::Program::build
    /// \brief   This function returns a matching kernel from a source code
    /// \details This function will first read the provided file for source code,
    ///          and will then build the provided source code and will
    ///          finally return the matching kernel.
    /// \see     GetProgramFromFile()
    ///
    cl_int GetKernelFromFile(std::ifstream & File, const char * KernelName,
                             cl::Kernel & Kernel) {
        cl::Program Program;
        cl_int Error = GetProgramFromFile(File, Program);
        if (Error !=  CL_SUCCESS) {
            return Error;
        }

        return GetKernelFromProgram(Program, KernelName, Kernel);
    }

    ///
    /// \fn      GetKernelFromSource
    /// \param   Source     Program source code that contains the kernel
    /// \param   KernelName The name of the kernel
    /// \param   Kernel     The matching kernel
    /// \return  Any of the OpenCL error of cl::Kernel, cl::Program and cl::Program::build
    /// \brief   This function returns a matching kernel from a source code
    /// \details This function will first build the provided source code and will
    ///          then return the matching kernel.
    /// \see     GetProgramFromSource()
    ///
    cl_int GetKernelFromSource(const std::string & Source,
                               const char * KernelName, cl::Kernel & Kernel) {
        cl::Program Program;
        cl_int Error = GetProgramFromSource(Source, Program);
        if (Error !=  CL_SUCCESS) {
            return Error;
        }

        return GetKernelFromProgram(Program, KernelName, Kernel);
    }

    ///
    /// \fn      GetKernelFromSource
    /// \param   Source     Program source code that contains the kernel
    /// \param   Length     Length of the program source code
    /// \param   KernelName The name of the kernel
    /// \param   Kernel     The matching kernel
    /// \return  Any of the OpenCL error of cl::Kernel, cl::Program and cl::Program::build
    /// \brief   This function returns a matching kernel from a source code
    /// \details This function will first build the provided source code and will
    ///          then return the matching kernel.
    /// \see     GetProgramFromSource()
    ///
    cl_int GetKernelFromSource(const char * Source, size_t Length,
                               const char * KernelName, cl::Kernel & Kernel) {
        cl::Program Program;
        cl_int Error = GetProgramFromSource(Source, Length, Program);
        if (Error !=  CL_SUCCESS) {
            return Error;
        }

        return GetKernelFromProgram(Program, KernelName, Kernel);
    }

    ///
    /// \fn     GetKernelFromProgram
    /// \param  Program    The program that contains the kernel
    /// \param  KernelName The name of the kernel
    /// \param  Kernel     The matching kernel
    /// \return Any of the OpenCL error of cl::Kernel
    /// \brief  This function returns a matching kernel from a program
    ///
    cl_int GetKernelFromProgram(const cl::Program & Program,
                                const char * KernelName, cl::Kernel & Kernel) {
        cl_int Error;
        Kernel = cl::Kernel(Program, KernelName, &Error);
        return Error;
    }

    ///
    /// \fn      GetLastElapsedTime
    /// \param   ElapsedTime The elapsed time of the last event in ns
    /// \return  Any of the OpenCL error of cl::Event::getProfilingInfo
    /// \brief   This function returns the elapsed time of the last event
    /// \details Elasped time is taken into account between the start of the
    ///          command and its end.
    ///
    cl_int GetLastElapsedTime(double * ElapsedTime) {
        cl_int Error;
        double Start, End;

        Error = mEvent.getProfilingInfo(CL_PROFILING_COMMAND_START, &Start);
        if (Error != CL_SUCCESS) {
            return Error;
        }

        Error = mEvent.getProfilingInfo(CL_PROFILING_COMMAND_END, &End);
        if (Error != CL_SUCCESS) {
            return Error;
        }

        *ElapsedTime = (End - Start);

        return CL_SUCCESS;
    }

    ///
    /// \fn      ExecuteKernelFromFile
    /// \tparam  Args       Types of the kernel arguments
    /// \param   FileName   File that contains the kernel to execute
    /// \param   KernelName Kernel to execute
    /// \param   DataSize   Size on which the kernel will work
    /// \param   KernelArgs Arguments of the kernel
    /// \return  Any error code of OpenCL
    /// \brief   This function will execute a specific kernel from a source file
    /// \details The function will first open the file and read the source code.
    ///          Then, the source code will be built. Finally, the kernel will
    ///          be executed on the target device with the given arguments.
    ///
    template<typename... Args>
    cl_int ExecuteKernelFromFile(const char * FileName, const char * KernelName,
                                 long DataSize, const Args&... KernelArgs) {
        cl::Kernel Kernel;
        cl_int Error = GetKernelFromFile(FileName, KernelName, Kernel);
        if (Error != CL_SUCCESS)
        {
            return Error;
        }

        return ExecuteKernelFromKernelEx(Kernel, DataSize, 0, KernelArgs...);
    }

    ///
    /// \fn      ExecuteKernelFromFile
    /// \tparam  Args       Types of the kernel arguments
    /// \param   File       File that contains the kernel to execute
    /// \param   KernelName Kernel to execute
    /// \param   DataSize   Size on which the kernel will work
    /// \param   KernelArgs Arguments of the kernel
    /// \return  Any error code of OpenCL
    /// \brief   This function will execute a specific kernel from a source file
    /// \details The function will first read the file for the source code.
    ///          Then, the source code will be built. Finally, the kernel will
    ///          be executed on the target device with the given arguments.
    ///
    template<typename... Args>
    cl_int ExecuteKernelFromFile(std::ifstream & File, const char * KernelName,
                                 long DataSize, const Args&... KernelArgs) {
        cl::Kernel Kernel;
        cl_int Error = GetKernelFromFile(File, KernelName, Kernel);
        if (Error != CL_SUCCESS)
        {
            return Error;
        }

        return ExecuteKernelFromKernelEx(Kernel, DataSize, 0, KernelArgs...);
    }

    ///
    /// \fn      ExecuteKernelFromSource
    /// \tparam  Args       Types of the kernel arguments
    /// \param   Source     Source code that contains the kernel to execute
    /// \param   KernelName Kernel to execute
    /// \param   DataSize   Size on which the kernel will work
    /// \param   KernelArgs Arguments of the kernel
    /// \return  Any error code of OpenCL
    /// \brief   This function will execute a specific kernel from a source code
    /// \details The function will first build the source code. Then, the kernel
    ///          will be executed on the target device with the given arguments.
    ///
    template<typename... Args>
    cl_int ExecuteKernelFromSource(const std::string & Source,
                                   const char * KernelName, long DataSize,
                                   const Args&... KernelArgs) {
        cl::Kernel Kernel;
        cl_int Error = GetKernelFromSource(Source, KernelName, Kernel);
        if (Error != CL_SUCCESS)
        {
            return Error;
        }

        return ExecuteKernelFromKernelEx(Kernel, DataSize, 0, KernelArgs...);
    }

    ///
    /// \fn      ExecuteKernelFromSource
    /// \tparam  Args       Types of the kernel arguments
    /// \param   Source     Source code that contains the kernel to execute
    /// \param   Length     Length of the provided source code
    /// \param   KernelName Kernel to execute
    /// \param   DataSize   Size on which the kernel will work
    /// \param   KernelArgs Arguments of the kernel
    /// \return  Any error code of OpenCL
    /// \brief   This function will execute a specific kernel from a source code
    /// \details The function will first build the source code. Then, the kernel
    ///          will be executed on the target device with the given arguments.
    ///
    template<typename... Args>
    cl_int ExecuteKernelFromSource(const char * Source, size_t Length,
                                   const char * KernelName, long DataSize,
                                   const Args&... KernelArgs) {
        cl::Kernel Kernel;
        cl_int Error = GetKernelFromSource(Source, Length, KernelName, Kernel);
        if (Error != CL_SUCCESS)
        {
            return Error;
        }

        return ExecuteKernelFromKernelEx(Kernel, DataSize, 0, KernelArgs...);
    }

    ///
    /// \fn      ExecuteKernelFromProgram
    /// \tparam  Args       Types of the kernel arguments
    /// \param   Program    Program that contains the kernel to execute
    /// \param   KernelName Kernel to execute
    /// \param   DataSize   Size on which the kernel will work
    /// \param   KernelArgs Arguments of the kernel
    /// \return  Any error code of OpenCL
    /// \brief   This function will execute a specific kernel from a program
    /// \details The kernel will be executed on the target device with the given
    ///          arguments.
    ///
    template<typename... Args>
    cl_int ExecuteKernelFromProgram(const cl::Program & Program,
                                    const char * KernelName, long DataSize,
                                    const Args&... KernelArgs) {
        cl::Kernel Kernel;
        cl_int Error = GetKernelFromProgram(Program, KernelName, Kernel);
        if (Error != CL_SUCCESS)
        {
            return Error;
        }

        return ExecuteKernelFromKernelEx(Kernel, DataSize, 0, KernelArgs...);
    }

    ///
    /// \fn      ExecuteKernelFromKernel
    /// \tparam  Args       Types of the kernel arguments
    /// \param   Kernel     Kernel to execute
    /// \param   DataSize   Size on which the kernel will work
    /// \param   KernelArgs Arguments of the kernel
    /// \return  Any error code of OpenCL
    /// \brief   This function will execute a specific kernel from a program
    /// \details The kernel will be executed on the target device with the given
    ///          arguments.
    ///
    template<typename... Args>
    cl_int ExecuteKernelFromKernel(const cl::Kernel & Kernel, long DataSize,
                                   const Args&... KernelArgs) {
        cl::Kernel intKernel = Kernel;
        return ExecuteKernelFromKernelEx(intKernel, DataSize, 0, KernelArgs...);
    }

    ///
    /// \fn     ReadBuffer
    /// \tparam T      Type of the buffer elements
    /// \param  Buffer The buffer to read from device
    /// \param  Host   The buffer in which copy read elements
    /// \param  Size   Number of elements to read
    /// \return Any OpenCL error code from cl::Queue::enqueueReadBuffer
    /// \brief  The function will read a device buffer into a host buffer
    ///
    template<typename T>
    cl_int ReadBuffer(cl::Buffer & Buffer, T * Host, size_t Size) {
        INIT(Queue);

        assert(mDevices != 0);
        assert(mContext != 0);
        assert(mQueue != 0);

        return mQueue->enqueueReadBuffer(Buffer, true, 0, sizeof(T) * Size, Host,
                                         0, &mEvent);
    }

    ///
    /// \fn      SetParameter
    /// \param   Parameter The parameter to set
    /// \param   Value     The value of the parameter to set
    /// \return  CL_SUCCESS, CL_INVALID_OPERATION
    /// \brief   This function tries to define a parameter with the given value
    /// \warning TargetDevice parameter can only be set if no device was selected
    ///
    cl_int SetParameter(OpenCLParameters Parameter, unsigned long Value) {
        cl_int Error = CL_INVALID_OPERATION;

        switch (Parameter) {
            case TargetDevice:
                if (mDevices == 0) {
                    if (Value == CL_DEVICE_TYPE_ALL ||
                        Value <= (CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_CPU |
                                  CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR)) {
                        mTargetDevice = Value;
                        Error = CL_SUCCESS;
                    }
                }
                break;

            case MaxParameters:
            default:
                break;
        }

        return Error;
    }

    ///
    /// \fn      SetParameter
    /// \param   Parameter The parameter to set
    /// \param   Value     The value of the parameter to set
    /// \return  CL_SUCCESS, CL_INVALID_OPERATION
    /// \brief   This function tries to define a parameter with the given value
    /// \warning TargetDevice parameter can only be set if no device was selected
    ///
    cl_int SetParameter(OpenCLParameters Parameter, std::string & Value) {
        cl_int Error = CL_INVALID_OPERATION;

        switch (Parameter) {
            case BuildOptions:
                mBuildOptions = Value;
                Error = CL_SUCCESS;
                break;

            case MaxParameters:
            default:
                break;
        }

        return Error;
    }

    ///
    /// \fn     WaitForLastEvent
    /// \return Any error code of cl::Event::wait
    /// \brief  This function will block the caller until the event is done
    ///
    cl_int WaitForLastEvent() {
        return mEvent.wait();
    }

    ///
    /// \fn     WriteBuffer
    /// \tparam T      Type of the buffer elements
    /// \param  Buffer The buffer to write on device
    /// \param  Host   The buffer from which read elements
    /// \param  Size   Number of elements to write
    /// \return Any OpenCL error code from cl::Queue::enqueueWriteBuffer
    /// \brief  The function will write a host buffer into a device buffer
    ///
    template<typename T>
    cl_int WriteBuffer(cl::Buffer & Buffer, T * Host, size_t Size) {
        INIT(Queue);

        assert(mDevices != 0);
        assert(mContext != 0);
        assert(mQueue != 0);

        return mQueue->enqueueWriteBuffer(Buffer, true, 0, sizeof (T) * Size, Host,
                                          0, &mEvent);
    }
};
}
