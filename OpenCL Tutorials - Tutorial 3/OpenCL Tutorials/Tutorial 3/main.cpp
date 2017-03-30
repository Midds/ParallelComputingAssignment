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
		// host operations
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

		std::vector<mytype> A;
		// reading in the values from file
		ifstream file;
		file.open("../../temp_lincolnshire_datasets/temp_lincolnshire.txt"); // open io stream
		string output;
		string sub;

		if (file.is_open()) {
			while (!file.eof()) {
				while (getline(file, output))
				{
					// for each line get the value 17 characters from the first " " in the txt file. This will always be the temperature value.
					sub = output.substr(output.find(" ") + 17);
					// insert the value into a vector
					std::vector<mytype> A_ext(1, std::stof(sub));
					A.insert(A.end(), A_ext.begin(), A_ext.end());
				}
			}
		}

		file.close(); // close io stream
		
		std::cout << "File read in complete...\n" << std::endl;

		//host - input

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 22;
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
		std::vector<mytype> B(nr_groups); // for min val
		std::vector<mytype> C(nr_groups); // for max val
		std::vector<mytype> D(nr_groups); // for avg
		std::vector<mytype> E(nr_groups); // for std dev
		std::vector<mytype> F(nr_groups); // for std dev

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size); // input vector
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size); // for min val
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size); // for max val
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size); // for avg
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, output_size); // for std dev
		cl::Buffer buffer_F(context, CL_MEM_READ_WRITE, output_size); // for std dev

		// device - operations
		// copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_E, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_F, 0, 0, output_size);//zero B buffer on device memory

		// initialise some variables needed for the upcoming kernals
		size_t workGroups = nr_groups*sizeof(mytype);
		float minElements = output_size;
		float finalMin, finalMax, finalAvg, finalStdDev = 0;
		float b_padding = 0;
		std::vector<mytype> originA = A; // needed to reset A after each loop
		
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
		
		// create profiling event and some variables to hold the execution times and memory transfer times for each loop iteration
		cl::Event prof_event;
		double kernalTime = 0;
		string memoryTransferTime;
		string subs;

		// Minimum
		while (minElements > local_size) {
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
			// launching kernal with profiling event
			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, workGroups, &B[0]); // Copy the result from device to host

			// resize B to the number of workgroups
			B.resize(nr_groups);
			// resize nr_groups so the next iteration will keep lowering the size of B with each B.resize()
			nr_groups = (B.size() / local_size);
			// Update minElements with the new size so the loop won't go on indefinitely 
			minElements = B.size();
			// Finally copy the final vector B into A - ready to be sent to the device in the next iteration
			A = B;

			// Add the kernal execution time for the last kernal execution to the total time so far
			kernalTime = kernalTime + (prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
			// get memory transfer time for each loop
			memoryTransferTime = GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US);
			std::cout << memoryTransferTime << std::endl;
		}
		// store the min in finalMin
		finalMin = B[0];
		// serially loop through the final x values for the min, where x = local_size
		// this is needed as the above loop stops when the number of elements gets lower than local_size
		for (int i = 1; i < B.size(); i++) {
			if (finalMin > B[i])
				finalMin = B[i];
		}
		
		std::cout << "Minimum Kernel - execution time [Microseconds]: " << kernalTime / 1000 << std::endl;
		std::cout << "Min = " << finalMin << "\n" << std::endl;

		// reset some variables needed for the next kernal
		nr_groups = input_elements / local_size;
		minElements = output_size;
		A = originA;
		kernalTime = 0;

		// Maximum
		while (minElements > local_size) {
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
			queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, workGroups, &C[0]);
			
			// resize C to the number of workgroups
			C.resize(nr_groups);
			// resize nr_groups so the next iteration will keep lowering the size of C with each C.resize()
			nr_groups = C.size() / local_size;
			// Update minElements with the new size so the loop won't go on indefinitely 
			minElements = C.size();
			// Finally copy the final vector C into A - ready to be sent to the device in the next iteration
			A = C;

			// Add the kernal execution time for the last kernal execution to the total time so far
			kernalTime = kernalTime + (prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
			// get memory transfer time for each loop
			memoryTransferTime = GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US);
			std::cout << memoryTransferTime << std::endl;
		}
		
		// store the max in finalMax
		finalMax = C[0];

		// serially loop through the final x values for the max, where x = local_size
		// this is needed as the above loop stops when the number of elements gets lower than local_size
		for (int i = 1; i < C.size(); i++) {
			if (finalMax < C[i])
				finalMax = C[i];
		}
		
		std::cout << "Maximum Kernel - execution time [Microseconds]: " << kernalTime / 1000 << std::endl;
		std::cout << "Max = " << finalMax << "\n" << std::endl;

		// reset some variables needed for the next kernal
		nr_groups = input_elements / local_size;
		minElements = output_size;
		A = originA;
		kernalTime = 0;

		// Average
		while (minElements > local_size) {
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
			queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, workGroups, &D[0]);
			
			// resize D to the number of workgroups
			D.resize(nr_groups);

			// pad the vector with 0s if the size of the array isn't a multiple of local_size
			// these zeros won't effect the final averaging sum
			b_padding = D.size() % local_size;
			if (b_padding) {
				//create an extra vector with neutral values
				std::vector<mytype> D_ext((local_size - b_padding), 0);
				//append that extra vector to D
				D.insert(D.end(), D_ext.begin(), D_ext.end());
			}

			// resize nr_groups so the next iteration will keep lowering the size of C with each C.resize()
			nr_groups = D.size() / local_size;
			// Update minElements with the new size so the loop won't go on indefinitely 
			minElements = D.size();
			// Finally copy the final vector D into A - ready to be sent to the device in the next iteration
			A = D;

			// Add the kernal execution time for the last kernal execution to the total time so far
			kernalTime = kernalTime + (prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
			// get memory transfer time for each loop
			memoryTransferTime = GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US);
			std::cout << memoryTransferTime << std::endl;
		}
		
		// store the average in finalAvg
		finalAvg = D[0];

		// serially loop through the final x values for the min, where x = local_size
		// this is needed as the above loop stops when the number of elements gets lower than local_size
		for (int i = 1; i < D.size(); i++) {
				finalAvg = finalAvg + D[i];
		}

		// divide the final sum by the original number of input elements to get the final average
		// important to use origin_input_elements not input_elements so the padding won't effect the result
		finalAvg = finalAvg / origin_input_elements;

		std::cout << "Average Kernel - execution time [Microseconds]: " << kernalTime / 1000 << std::endl;
		std::cout << "Avg = " << finalAvg << "\n" << std::endl;

		// Standard deviation														 
		cl::Kernel kernel_4 = cl::Kernel(program, "std_dev");
		kernel_4.setArg(0, buffer_A); // input
		kernel_4.setArg(1, buffer_E); // output
		kernel_4.setArg(2, finalAvg); // mean
		kernel_4.setArg(3, cl::Local(local_size * sizeof(mytype)));//local memory size
		
		// reset some variables needed for the next kernal
		nr_groups = input_elements / local_size;
		minElements = output_size;
		A = originA;
		kernalTime = 0;


		// std dev
		while (minElements > local_size) {
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);

			queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, workGroups, &E[0]);

			E.resize(workGroups);
			workGroups = E.size() / local_size;
			//std::cout << "A = " << A << std::endl;
			//std::cout << "E = " << E << std::endl;
			minElements = 1;

			//printf("C size = %d \n", C.size());
			//workGroups = nr_groups * sizeof(mytype);
			A = E;

			// Add the kernal execution time for the last kernal execution to the total time so far
			kernalTime = kernalTime + (prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
			// get memory transfer time for each - as this is retruned in a string will need to get substring to add up each seperately
			memoryTransferTime = GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US);
			std::cout << memoryTransferTime << std::endl;
		}

		// reset some variables needed for the next kernal
		// but dont reset kernal time
		nr_groups = input_elements / local_size;
		minElements = output_size;
		A = E;

		// Average -- Need to uncomment the line below that divides answer
		cl::Kernel kernel_5 = cl::Kernel(program, "avg");
		kernel_5.setArg(0, buffer_A);
		kernel_5.setArg(1, buffer_F);
		kernel_5.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		// summing up
		while (minElements > local_size) {
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
			queue.enqueueNDRangeKernel(kernel_5, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_F, CL_TRUE, 0, workGroups, &F[0]);

			// resize D to the number of workgroups
			F.resize(nr_groups);

			// pad the vector with 0s if the size of the array isn't a multiple of local_size
			// these zeros won't effect the final averaging sum
			b_padding = F.size() % local_size;
			if (b_padding) {
				//create an extra vector with neutral values
				std::vector<mytype> F_ext((local_size - b_padding), 0);
				//append that extra vector to D
				F.insert(F.end(), F_ext.begin(), F_ext.end());
			}

			// resize nr_groups so the next iteration will keep lowering the size of C with each C.resize()
			nr_groups = F.size() / local_size;
			// Update minElements with the new size so the loop won't go on indefinitely 
			minElements = F.size();
			// Finally copy the final vector D into A - ready to be sent to the device in the next iteration
			A = F;

			// Add the kernal execution time for the last kernal execution to the total time so far
			kernalTime = kernalTime + (prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
			// get memory transfer time for each - as this is retruned in a string will need to get substring to add up each seperately
			memoryTransferTime = GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US);
			std::cout << memoryTransferTime << std::endl;
		}
		// store the average in finalAvg
		finalAvg = F[0];

		// serially loop through the final x values for the min, where x = local_size
		// this is needed as the above loop stops when the number of elements gets lower than local_size
		for (int i = 1; i < F.size(); i++) {
			finalAvg = finalAvg + F[i];
		}

		// divide the final sum by the original number of input elements -1 to get the variance
		// std::cout << "Sum = " << finalAvg << std::endl;				
		finalStdDev = (finalAvg / (input_elements - 1));
		//std::cout << "Variance = " << finalStdDev << std::endl;

		finalStdDev = sqrt(finalStdDev);

		std::cout << "Standard deviation Kernel - execution time [Microseconds]: " << kernalTime /1000 << std::endl;
		std::cout << "Standard Deviation = " << finalStdDev << std::endl;

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
