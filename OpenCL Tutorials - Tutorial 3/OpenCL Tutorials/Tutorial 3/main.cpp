// TODO - fix padding
// - read in from files small/large
// time profiling - add up in loops 
// test
// comment
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

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

		typedef float mytype;
		
		//std::vector<mytype> A {6, 2, 1, 5, 2, 6, 13, 10, 13, 9, 4, -2, 3, 2, 2, 3, 7, 9, 9, 6, 3, 4, 3, 1, 4, 9, 8, 14, 15};
		std::vector<mytype> A;

		ifstream myReadFile;
		myReadFile.open("../../temp_lincolnshire_datasets/temp_lincolnshire_short.txt");
		string output;
		string file_contents;
		string sub;

		if (myReadFile.is_open()) {
			while (!myReadFile.eof()) {
				while (getline(myReadFile, output))
				{
					sub = output.substr(output.find(" ") + 17);
					
					std::vector<mytype> A_ext(1, std::stof(sub));
					A.insert(A.end(), A_ext.begin(), A_ext.end());
					
					file_contents += sub;
					file_contents.push_back('\n');
				}
			}
		}

		myReadFile.close();
		////cout << file_contents << endl;
		
		//Part 4 - memory allocation
		//host - input
		//std::vector<mytype> A { 5.1, -6.3, 5.2, -1.5, 13.3, -7.6, 21.8, -5.9, 1.4, 9.1, 1.1, 2.4, 14.1, 14.2, 52.1, 16.4, -5.3, 6.7, 8.9, 3.2 };//allocate 10 elements
		//std::vector<mytype> A{ 5.1, -6.3, -14.9, -1.5, -13.3, -7.6, 21.8, -5.9, 9.1 ,  1.4, - 91 };//allocate 10 elements

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 5;
		size_t padding_size = A.size() % local_size;
		size_t origin_input_elements = A.size(); // need the original size before padding is added

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<mytype> A_ext((local_size-padding_size), 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

	

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;
		size_t output_size = A.size() * sizeof(mytype);//size in bytes

		//host - output
		//std::vector<mytype> B(input_elements);
		//changes number of elements to 1
		std::vector<mytype> B(nr_groups); // for min val
		std::vector<mytype> C(nr_groups); // for max val
		std::vector<mytype> D(nr_groups); // for avg
		std::vector<mytype> E(nr_groups); // for std dev

		//std::vector<mytype> D((int)nr_groups);
		//std::vector<mytype> E(10);
		
		//std::vector<mytype> H((int)nr_bins);
		//std::vector<mytype> H((int)nr_groups);
		
		//size_t output_hist_size = H.size()*sizeof(mytype);
		//cl::Buffer buffer_H(context, CL_MEM_READ_WRITE, output_hist_size);
		//queue.enqueueFillBuffer(buffer_H, 0, 0, output_hist_size);


		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size); // input vector
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size); // for min val
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size); // for max val
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size); // for avg
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, output_size); // for std dev


		// device operations

		// copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		// Setup and execute all kernels (i.e. device code)
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



		

		

		//// Average -- Need to uncomment the line below that divides answer
		//cl::Kernel kernel_3 = cl::Kernel(program, "avg");
		//kernel_3.setArg(0, buffer_A);
		//kernel_3.setArg(1, buffer_D);
		//kernel_3.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		
		size_t reduct_output_size = nr_groups; // needed to stop the program crashing //delete
		size_t workGroups = nr_groups*sizeof(mytype);
		float minElements = output_size;
		float finalMin, finalMax, finalAvg, finalStdDev = 0;
		float b_padding = 0;
		std::vector<mytype> tempB(nr_groups); // for min val

		//std::cout << "A = " << A << std::endl;

		/*std::cout << "A[0] = " << A[0] << std::endl;
		std::cout << "A size = " << A.size() << std::endl;
		std::cout << "A[size-1] = " << A[A.size()-1] << std::endl;
		std::cout << "A[size-2] = " << A[A.size() - 2] << std::endl;
		std::cout << "A[size-3] = " << A[A.size() - 3] << std::endl;
		std::cout << "A[size-4] = " << A[A.size() - 4] << std::endl;
		std::cout << "output size = " << output_size << std::endl;
		std::cout << "nr groups = " << nr_groups << std::endl;*/
		//std::cout << "-25 = " << A[3292] << std::endl;


		cl::Event prof_event;
		
		// Minimum
		cl::Kernel kernel_1 = cl::Kernel(program, "min_val");		
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		// Maximum
		cl::Kernel kernel_2 = cl::Kernel(program, "max_val");
		kernel_2.setArg(0, buffer_A);
		kernel_2.setArg(1, buffer_C);
		kernel_2.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		// Average -- Need to uncomment the line below that divides answer
		cl::Kernel kernel_3 = cl::Kernel(program, "avg");
		kernel_3.setArg(0, buffer_A);
		kernel_3.setArg(1, buffer_D);
		kernel_3.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		
		std::cout << "B size before = " << B.size() << std::endl;

		// Minimum
		while (minElements > local_size) {
			//std::cout << "Sending A" << A << std::endl;
			//std::cout << "Sending B" << B << std::endl;
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);

			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, workGroups, &B[0]); // Copy the result from device to host
			//std::cout << "Immediate B = " << B << std::endl;

			B.resize(nr_groups);

			b_padding = B.size() % local_size;
			if (b_padding) {
				//create an extra vector with neutral values
				std::vector<mytype> B_ext((local_size - b_padding), 0);
				//append that extra vector to our input
				B.insert(B.end(), B_ext.begin(), B_ext.end());
			}

			nr_groups = (B.size() / local_size);

			std::cout << "" << std::endl;
			//std::cout << "B = " << B << std::endl;
			std::cout << "B size= " << B.size() << std::endl;
			std::cout << "NR_groups = " << nr_groups << std::endl;
			std::cout << std::endl;

			//std::cout << "A = " << A << std::endl;
			minElements = B.size();
					
			//std::cout << "A = " << A << std::endl;

			A = B;
			//std::cout << "A after = " << A << std::endl;

			//printf("B size = %d \n", B.size());
			//workGroups = nr_groups * sizeof(mytype);
		}
		std::cout << "B after loop = " << B << std::endl;
		
		finalMin = B[0];

		// serially loop through the final x values for the min, where x = local_size
		for (int i = 1; i < B.size(); i++) {
			if (finalMin > B[i])
				finalMin = B[i];
		}
		std::cout << "Min = " << finalMin << std::endl;
		
		nr_groups = input_elements / local_size;
		minElements = output_size;

		// Maximum
		while (minElements > local_size) {

			queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, workGroups, &C[0]);

			C.resize(nr_groups);
			nr_groups = C.size() / local_size;
			//std::cout << "A = " << A << std::endl;
			//std::cout << "Max = " << C << std::endl;
			minElements = C.size();

			//printf("C size = %d \n", C.size());
			//workGroups = nr_groups * sizeof(mytype);
		}
		//std::cout << "C after loop = " << C << std::endl;
		finalMax = C[0];

		// serially loop through the final x values for the min, where x = local_size
		for (int i = 1; i < C.size(); i++) {
			if (finalMax < C[i])
				finalMax = C[i];
		}
		std::cout << "Max = " << finalMax << std::endl;	
		
		nr_groups = input_elements / local_size;
		minElements = output_size;

		//std::cout << "D before = " << D << std::endl;

		// avg
		while (minElements > local_size) {

			queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, workGroups, &D[0]);

			D.resize(nr_groups);
			nr_groups = D.size() / local_size;
			//std::cout << "A = " << A << std::endl;
			//std::cout << "D = " << D << std::endl;
			minElements = D.size();

			//printf("C size = %d \n", C.size());
			//workGroups = nr_groups * sizeof(mytype);
		}
		std::cout << "D after loop = " << D << std::endl;
		
		finalAvg = D[0];

		// serially loop through the final x values for the min, where x = local_size
		for (int i = 1; i < D.size(); i++) {
				finalAvg = finalAvg + D[i];
		}
		finalAvg = finalAvg / origin_input_elements;
		std::cout << "Avg = " << finalAvg << std::endl;
		std::cout << "" << std::endl;



		// Standard deviation														 
		cl::Kernel kernel_4 = cl::Kernel(program, "std_dev");
		kernel_4.setArg(0, buffer_A); // input
		kernel_4.setArg(1, buffer_E); // output
		kernel_4.setArg(2, finalAvg); // mean
		kernel_4.setArg(3, cl::Local(local_size * sizeof(mytype)));//local memory size
		
		nr_groups = input_elements / local_size;
		minElements = output_size;


		// std dev
		while (minElements > local_size) {

			queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, workGroups, &E[0]);

			E.resize(nr_groups);
			nr_groups = E.size() / local_size;
			//std::cout << "A = " << A << std::endl;
			//std::cout << "E = " << E << std::endl;
			minElements = E.size();

			//printf("C size = %d \n", C.size());
			//workGroups = nr_groups * sizeof(mytype);
		}
		//std::cout << "E after loop = " << E << std::endl;

		finalStdDev = E[0];

		// serially loop through the final x values for the min, where x = local_size
		for (int i = 1; i < E.size(); i++) {
			finalStdDev = finalStdDev + E[i];
		}
	
		finalStdDev = (finalStdDev / (input_elements - 1));
		std::cout << "Variance = " << finalStdDev << std::endl;

		finalStdDev = sqrt(finalStdDev);
		std::cout << "Standard Deviation = " << finalStdDev << std::endl;

		std::cout << "\nKernel execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;


		//string a = "";
		//getline(cin, a);



	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
