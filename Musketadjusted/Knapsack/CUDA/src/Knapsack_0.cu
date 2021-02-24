#include <cuda.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <array>
#include <vector>
#include <sstream>
#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <memory>
#include <cstddef>
#include <type_traits>
#include <fstream>
#include <sstream>


#include "../include/musket.cuh"
#include "../include/Knapsack_0.cuh"
	
	
	
	const int global_object_values = 6;
	const int global_n_constraints = 10;
	const double global_Q = 0.0;

#define TAUMAX 2

	
	struct InitPheros_map_index_in_place_array_functor{
		
		InitPheros_map_index_in_place_array_functor(){}
		
		~InitPheros_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int i, double y){
            double randn = curand_uniform(&d_rand_states_ind[1]);
            return randn * TAUMAX;
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
        curandState* d_rand_states_ind;
		
	};
	struct InitFreeSpace_map_index_in_place_array_functor{
		
		InitFreeSpace_map_index_in_place_array_functor(const mkt::DArray<int>& _constraint_max_values) : constraint_max_values(_constraint_max_values){}
		
		~InitFreeSpace_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int i, int y){
			int j = (static_cast<int>((i)) % (n_ants));
			return constraint_max_values.get_global((j))/* TODO: For multiple GPUs*/;
		}
	
		void init(int device){
			constraint_max_values.init(device);
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		int n_ants;
		
		mkt::DeviceArray<int> constraint_max_values;
	};
	struct Generate_solutions_map_index_in_place_array_functor{
		
		Generate_solutions_map_index_in_place_array_functor(const mkt::DArray<int>& _d_ant_available_objects, const mkt::DArray<int>& _object_values, const mkt::DArray<double>& _d_pheromones, const mkt::DArray<int>& _dimensions_values, const mkt::DArray<int>& _d_free_space, const mkt::DArray<double>& _d_eta, const mkt::DArray<double>& _d_tau, const mkt::DArray<double>& _d_probabilities, const mkt::DArray<int>& _d_ant_solutions, const mkt::DArray<int>& _constraint_max_values) : d_ant_available_objects(_d_ant_available_objects), object_values(_object_values), d_pheromones(_d_pheromones), dimensions_values(_dimensions_values), d_free_space(_d_free_space), d_eta(_d_eta), d_tau(_d_tau), d_probabilities(_d_probabilities), d_ant_solutions(_d_ant_solutions), constraint_max_values(_constraint_max_values){}
		
		~Generate_solutions_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int i, int value){
			int ant_index = (i);
			int value_object_j = 0;
			double pheromone_to_object_j = 0.0;
			int size_i_object_j = 0;
			double average_tightness_object_j = 0.0;
			int free_space_i = 0;
			double eta = 0.0;
			double tau = 0.0;
			double eta_tau_sum = 0.0;
			int fitness = 0;
			bool is_too_big = false;
			bool is_possible = false;
			int select_index = 0;
			for(int step = 0; ((step) < (d_n_objects)); step++){
				eta_tau_sum = 0.0;
				is_possible = false;
				for(int object_j = 0; ((object_j) < (d_n_objects)); object_j++){
					
					if((d_ant_available_objects.get_global((((ant_index) * (d_n_objects)) + (object_j)))/* TODO: For multiple GPUs*/ == 1)){
                        value_object_j = object_values.get_global((object_j))/* TODO: For multiple GPUs*/;
                        pheromone_to_object_j = d_pheromones.get_global((((step) * (d_n_objects)) + (object_j)))/* TODO: For multiple GPUs*/;
                        average_tightness_object_j = 0.0;
                        is_too_big = false;
                        for(int ii = 0; ((ii) < (d_n_constraints)); ii++){
                            size_i_object_j = dimensions_values.get_global((((i) * (d_n_objects)) + (object_j)))/* TODO: For multiple GPUs*/;
                            free_space_i = d_free_space.get_global((((ant_index) * (d_n_constraints)) + (i)))/* TODO: For multiple GPUs*/;

                            if(((size_i_object_j) <= (free_space_i))){

                                if(((free_space_i) == 0.0)){
                                    average_tightness_object_j = (1.0 + (average_tightness_object_j));
                                } else {
                                        average_tightness_object_j = (((size_i_object_j) / (free_space_i)) + (average_tightness_object_j));
                                }
                            } else {
								is_too_big = true;
							}
					    }
					
                        if(!(is_too_big)){
                            average_tightness_object_j = static_cast<double>(((average_tightness_object_j) / (d_n_constraints)));
                            eta = pow(((value_object_j) / (average_tightness_object_j)), 1);
                            tau = pow((pheromone_to_object_j), 1);
                            eta_tau_sum = (((eta) * (tau)) + (eta_tau_sum));
                            d_eta.set_global((((ant_index) * (d_n_objects)) + (object_j)), (eta));
                            d_tau.set_global((((ant_index) * (d_n_objects)) + (object_j)), (tau));
                            is_possible = true;
                        } else {
							d_eta.set_global((((ant_index) * (d_n_objects)) + (object_j)), 0.0);
							d_tau.set_global((((ant_index) * (d_n_objects)) + (object_j)), 0.0);
						}
					}
					else {
                        d_eta.set_global((((ant_index) * (d_n_objects)) + (object_j)), 0.0);
                        d_tau.set_global((((ant_index) * (d_n_objects)) + (object_j)), 0.0);
					}
				}
				
				if((is_possible)){
                    for(int object_j = 0; ((object_j) < (d_n_objects)); object_j++){
                        d_probabilities.set_global((((ant_index) * (d_n_objects)) + (object_j)), ((d_eta.get_global((((ant_index) * (d_n_objects)) + (object_j)))/* TODO: For multiple GPUs*/ * d_tau.get_global((((ant_index) * (d_n_objects)) + (object_j)))/* TODO: For multiple GPUs*/) / (eta_tau_sum)));
                    }
                    double random =  curand_uniform(&d_rand_states_ind[ant_index]);
                    select_index = 0;
                    int selected_object = 0;
                    double sum = 0.0;
                    double prob = 0.0;
                    while ((sum <= random) && (select_index < d_n_objects)){
                        prob = d_probabilities.get_global((((ant_index) * (d_n_objects)) + (select_index)))/* TODO: For multiple GPUs*/;

                        if(((prob) > 0.0)){
                            sum = ((sum) + (prob));
                            selected_object = (select_index);
                        }
                        select_index = ((select_index) + 1);
                    }
                    d_ant_solutions.set_global((((ant_index) * (d_n_objects)) + (step)), (selected_object));
                    d_ant_available_objects.set_global((((ant_index) * (d_n_objects)) + (selected_object)), 0);
                    for(int j = 0; ((j) < (d_n_constraints)); j++){
                        d_free_space.set_global((((ant_index) * (d_n_constraints)) + (j)), (d_free_space.get_global((((ant_index) * (d_n_constraints)) + (j)))/* TODO: For multiple GPUs*/ - dimensions_values.get_global((((j) * (d_n_objects)) + (selected_object)))/* TODO: For multiple GPUs*/));
                    }
                    fitness = (object_values.get_global((selected_object))/* TODO: For multiple GPUs*/ + (fitness));
				} else {
				    d_ant_solutions.set_global((((ant_index) * (d_n_objects)) + (step)), 0);
				}
			}
			for(int j = 0; ((j) < (d_n_constraints)); j++){
				d_free_space.set_global((((ant_index) * (d_n_constraints)) + (j)), constraint_max_values.get_global((j))/* TODO: For multiple GPUs*/);
			}
			for(int j = 0; ((j) < (d_n_objects)); j++){
				d_ant_available_objects.set_global((((ant_index) * (d_n_objects)) + (j)), 1);
			}
			return (fitness);
		}
	
		void init(int device){
			d_ant_available_objects.init(device);
			object_values.init(device);
			d_pheromones.init(device);
			dimensions_values.init(device);
			d_free_space.init(device);
			d_eta.init(device);
			d_tau.init(device);
			d_probabilities.init(device);
			d_ant_solutions.init(device);
			constraint_max_values.init(device);
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		int d_n_objects;
		int d_n_constraints;
        curandState* d_rand_states_ind;

		mkt::DeviceArray<int> d_ant_available_objects;
		mkt::DeviceArray<int> object_values;
		mkt::DeviceArray<double> d_pheromones;
		mkt::DeviceArray<int> dimensions_values;
		mkt::DeviceArray<int> d_free_space;
		mkt::DeviceArray<double> d_eta;
		mkt::DeviceArray<double> d_tau;
		mkt::DeviceArray<double> d_probabilities;
		mkt::DeviceArray<int> d_ant_solutions;
		mkt::DeviceArray<int> constraint_max_values;
	};
	struct Evaporate_map_index_in_place_array_functor{
		
		Evaporate_map_index_in_place_array_functor(){}
		
		~Evaporate_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int i, double value){
			return ((value) * (evaporation));
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		double evaporation;
		
	};

    __global__ void setup_rand_kernel(curandState * state, unsigned long seed) {

        int id = blockIdx.x * blockDim.x + threadIdx.x;

        curand_init(seed, id, 0, &state[id]);

        __syncthreads();
    }

	int main(int argc, char** argv) {
		mkt::init();
		
		
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point complete_timer_start = std::chrono::high_resolution_clock::now();

        int n_objects = 0;
        int n_constraints = 0;
        int n_ants = 1024;
        int problem = 1;
        int Q = 0.5;

        switch (problem) {
            case 1:
                n_objects = 6;
                n_constraints = 10;
                break;
            case 2:
                n_objects = 10;
                n_constraints = 10;
                break;
            case 3:
                n_objects = 15;
                n_constraints = 10;
                break;
            case 4:
                n_objects = 20;
                n_constraints = 10;
                break;
            case 5:
                n_objects = 28;
                n_constraints = 10;
                break;
            case 6:
                n_objects = 39;
                n_constraints = 5;
                break;
            case 7:
                n_objects = 50;
                n_constraints = 5;
                break;
        }

        mkt::DArray<int> object_values(0, 6, 6, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
        mkt::DArray<int> dimensions_values(0, 60, 60, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
        mkt::DArray<int> constraints_max_values(0, 60, 60, 0, 1, 0, 0, mkt::DIST, mkt::COPY);

        std::string file_name = "mknap1";

        switch (problem) {
            case 1:
                file_name = "mknap1";
                break;
            case 2:
                file_name = "mknap2";;
                break;
            case 3:
                file_name = "mknap3";
                break;
            case 4:
                file_name = "mknap4";
                break;
            case 5:
                file_name = "mknap5";
                break;
            case 6:
                file_name = "mknap6";
                break;
            case 7:
                file_name = "mknap7";
                break;
        }

        std::ifstream inputFile(file_name);
        std::string sline;

        //header line ---- this line was already set manualy and therefore here ignored
        getline(inputFile, sline, '\n');
        std::istringstream linestream(sline);

        //Get Object values
        getline(inputFile, sline, '\n');
        std::istringstream linestream1(sline);
        for(int i = 0 ; i < n_objects ; i++){
            linestream1 >> object_values[i];
            Q += object_values[i];
        }

        //Get Constraint Values
        for(int i = 0 ; i < n_constraints ; i++){
            getline(inputFile, sline, '\n');
            std::istringstream linestream2(sline);
            for(int j = 0 ; j < n_objects ; j++){
                linestream2 >> dimensions_values[i*n_objects +j];
            }
        }

        getline(inputFile, sline, '\n');
        std::istringstream linestream3(sline);
        for(int i = 0 ; i < n_constraints ; i++){
            linestream3 >> constraints_max_values[i];
        }

        inputFile.close();

        Q = 1/Q;

        int antstimesobject = n_ants * n_objects;
        int objectssquared = n_objects * n_objects;
		mkt::DArray<int> d_ant_solutions(0, antstimesobject, antstimesobject, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> d_best_solution(0, object_values, object_values, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> d_ant_available_objects(0, antstimesobject, antstimesobject, 1, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<double> d_pheromones(0, objectssquared, objectssquared, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<double> d_delta_phero(0, objectssquared, objectssquared, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<double> d_probabilities(0, antstimesobject, antstimesobject, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> d_free_space(0, n_ants * n_constraints, n_ants * n_constraints, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<double> d_eta(0, antstimesobject, antstimesobject, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<double> d_tau(0, antstimesobject, antstimesobject, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> d_ant_fitness(0, n_ants, n_ants, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		
		InitPheros_map_index_in_place_array_functor initPheros_map_index_in_place_array_functor{};
		InitFreeSpace_map_index_in_place_array_functor initFreeSpace_map_index_in_place_array_functor{constraint_max_values};
		Generate_solutions_map_index_in_place_array_functor generate_solutions_map_index_in_place_array_functor{d_ant_available_objects, object_values, d_pheromones, dimensions_values, d_free_space, d_eta, d_tau, d_probabilities, d_ant_solutions, constraint_max_values};
		Evaporate_map_index_in_place_array_functor evaporate_map_index_in_place_array_functor{};

		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		int n_iterations = 5;
		double evaporation = 0.5;

		double best_fitness = 0.0;
		double mean_times = 0.0;
        curandState* d_rand_states_ind;
        cudaMalloc(&d_rand_states_ind, n_ants *  sizeof(curandState));

        setup_rand_kernel<<<n_blocks, block_size, 0, stream[0]>>>(d_rand_states_ind, time(NULL));


        initPheros_map_index_in_place_array_functor.d_rand_states_ind = d_rand_states_ind;
		mkt::map_index_in_place<double, InitPheros_map_index_in_place_array_functor>(d_pheromones, initPheros_map_index_in_place_array_functor);
		initFreeSpace_map_index_in_place_array_functor.n_ants = (n_ants);
		mkt::map_index_in_place<int, InitFreeSpace_map_index_in_place_array_functor>(d_free_space, initFreeSpace_map_index_in_place_array_functor);
		for(int ii = 0; ((ii) < (n_iterations)); ii++){
			generate_solutions_map_index_in_place_array_functor.d_n_objects = 6;
			generate_solutions_map_index_in_place_array_functor.d_n_constraints = 10;
			generate_solutions_map_index_in_place_array_functor.d_rand_states_ind = d_rand_states_ind;
			mkt::map_index_in_place<int, Generate_solutions_map_index_in_place_array_functor>(d_ant_fitness, generate_solutions_map_index_in_place_array_functor);
			for(int i = 0; ((i) < (n_ants)); i++){
				double ant_j_fitness = d_ant_fitness.get_global((i))/* TODO: For multiple GPUs*/;
				if(((ant_j_fitness) > (best_fitness))){
					best_fitness = (ant_j_fitness);
					for(int j = 0; ((j) < (n_objects)); j++){
						d_best_solution[j] = d_ant_solutions.get_global((((i) * (n_objects)) + (j)));
					}
				}
			}
			evaporate_map_index_in_place_array_functor.evaporation = (evaporation);
			mkt::map_index_in_place<double, Evaporate_map_index_in_place_array_functor>(d_pheromones, evaporate_map_index_in_place_array_functor);
		}
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point complete_timer_end = std::chrono::high_resolution_clock::now();
		double complete_seconds = std::chrono::duration<double>(complete_timer_end - complete_timer_start).count();
		printf("Complete execution time: %.5fs\n", complete_seconds);
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", 1);
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
