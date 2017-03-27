#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels3.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef double mytype;



		//Part 4 - memory allocation
		//host - input
		std::vector<mytype> A { 5.1, -6.3, 5.2, -1.5, 13.3, -7.6, 21.8, -5.9, 1.4, 9.1, 1.1, 2.4, 14.1, 14.2, 52.1, 16.4, -5.3, 6.7, 8.9, 3.2 };//allocate 10 elements
		//std::vector<mytype> A{1,1,1,3,-3,-1,1,1,1,1};
		//std::vector<mytype> A{ 5.1, -6.3, 5.2, -1.5, 13.3, -7.6, 21.8, -5.9, 1.4, 9.1 };//allocate 10 elements

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 5;

		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		//host - output
		//std::vector<mytype> B(input_elements);
		//changes number of elements to 1
		std::vector<mytype> B(nr_groups);
		//std::vector<mytype> C((int)nr_groups);
		//std::vector<mytype> D((int)nr_groups);
		//std::vector<mytype> E(10);
		
		//std::vector<mytype> H((int)nr_bins);
		//std::vector<mytype> H((int)nr_groups);
		
		//size_t output_hist_size = H.size()*sizeof(mytype);
		//cl::Buffer buffer_H(context, CL_MEM_READ_WRITE, output_hist_size);
		//queue.enqueueFillBuffer(buffer_H, 0, 0, output_hist_size);

		size_t output_size = A.size()*sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);


		//Part 5 - device operations

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		//5.2 Setup and execute all kernels (i.e. device code)
		//cl::Kernel kernel_1 = cl::Kernel(program, "scan_add");
		//kernel_1.setArg(0, buffer_A);
		//kernel_1.setArg(1, buffer_B);
		//kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		//kernel_1.setArg(3, cl::Local(local_size * sizeof(mytype)));//local memory size

		////call all kernels in a sequence
		//cl::Event prof_event;
		//queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);

		////5.3 Copy the result from device to host
		//queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		//cl::Kernel kernel_2 = cl::Kernel(program, "block_sum");
		//kernel_2.setArg(0, buffer_B);
		//kernel_2.setArg(1, buffer_C);
		//kernel_2.setArg(2, 5);//local memory size
		//queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(nr_groups), cl::NullRange, NULL, &prof_event);
		//queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);

		//cl::Kernel kernel_3 = cl::Kernel(program, "scan_add_atomic");
		//kernel_3.setArg(0, buffer_C);
		//kernel_3.setArg(1, buffer_D);
		//queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		//queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_size, &D[0]);

		//cl::Kernel kernel_4 = cl::Kernel(program, "scan_add_adjust");
		//kernel_4.setArg(0, buffer_E);
		//kernel_4.setArg(1, buffer_D);
		//queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		//queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, output_size, &E[0]);



		

		////// Maximum
		//cl::Kernel kernel_2 = cl::Kernel(program, "max_val");
		//kernel_2.setArg(0, buffer_A);
		//kernel_2.setArg(1, buffer_C);
		//kernel_2.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		//// Average -- Need to uncomment the line below that divides answer
		//cl::Kernel kernel_3 = cl::Kernel(program, "avg");
		//kernel_3.setArg(0, buffer_A);
		//kernel_3.setArg(1, buffer_D);
		//kernel_3.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		
		size_t reduct_output_size = nr_groups; // needed to stop the program crashing
		int workGroups = nr_groups*sizeof(mytype);
		int minElements = output_size;
		float finalMin = 0;


		cl::Event prof_event;
		std::cout << "A = " << A << std::endl;
		cl::Kernel kernel_1 = cl::Kernel(program, "min_val");
		
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
		printf("B size = %d \n", B.size());

		while (minElements > 2) {
			////// Minimum
			

			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, workGroups, &B[0]); // Copy the result from device to host
			A = B;
			B.resize(B.size()/2);
			std::cout << "A = " << A << std::endl;
			std::cout << "Min = " << B << std::endl;
			minElements = B.size();

			printf("B size = %d \n", B.size());
			cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, minElements);
			nr_groups = 2;
			workGroups = nr_groups * sizeof(mytype);

			if (local_size > B.size())
				local_size = 2;
		}
		if (B[0] > B[1])
			finalMin = B[1];
		else
			finalMin = B[0];
		std::cout << "Min = " << finalMin << std::endl;


		std::cout << "Kernel execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;

		/*queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);*/
		//queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		//queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_size, &D[0]);
		
		// For averaging
		//D[0] = D[0] / input_elements;

		// Standard deviation														 
		//cl::Kernel kernel_4 = cl::Kernel(program, "std_dev");
		//kernel_4.setArg(0, buffer_A); // input
		//kernel_4.setArg(1, buffer_E); // output
		//kernel_4.setArg(2, D[0]); // mean
		//kernel_4.setArg(3, cl::Local(local_size * sizeof(mytype)));//local memory size
		//queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		//queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, output_size, &E[0]);
		//E[0] = (E[0] / (input_elements - 1));

		
		/*std::cout << "Max = " << C << std::endl;
		std::cout << "Avg = " << D << std::endl;
		std::cout << "Std dev = " << E << std::endl;*/


		


		//string a = "";
		//getline(cin, a);



	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
