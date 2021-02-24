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
	
	
	#include "../include/musket.cuh"
	#include "../include/Knapsack_0.cuh"
	
	
	
	const int global_object_values = 6;
	const int global_n_constraints = 10;
	const double global_Q = 0.0;
	
	

	
	struct InitPheros_map_index_in_place_array_functor{
		
		InitPheros_map_index_in_place_array_functor(){}
		
		~InitPheros_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int i, double y){
			return 1.2;
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
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
						}
						 else {
								average_tightness_object_j = (((size_i_object_j) / (free_space_i)) + (average_tightness_object_j));
							}
						}
						 else {
								is_too_big = true;
							}
					}
					
					if(!(is_too_big)){
					average_tightness_object_j = static_cast<double>(((average_tightness_object_j) / (d_n_constraints)));
					eta = __powf(((value_object_j) / (average_tightness_object_j)), 1);
					tau = __powf((pheromone_to_object_j), 1);
					eta_tau_sum = (((eta) * (tau)) + (eta_tau_sum));
					d_eta.set_global((((ant_index) * (d_n_objects)) + (object_j)), (eta));
					d_tau.set_global((((ant_index) * (d_n_objects)) + (object_j)), (tau));
					is_possible = true;
					}
					 else {
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
				double random = 0.0;
				select_index = 0;
				int selected_object = 0;
				double sum = 0.0;
				double prob = 0.0;
				for(int i = 0; ((i) > 0); i++){
					prob = d_probabilities.get_global((((ant_index) * (d_n_objects)) + (select_index)))/* TODO: For multiple GPUs*/;
					
					if(((prob) > 0.0)){
					sum = ((sum) + (prob));
					selected_object = (select_index);
					}
					select_index = ((select_index) + 1);
					
					if(((sum) <= (random))){
					
					if(((select_index) < (d_n_objects))){
					i = -(1);
					}
					}
				}
				d_ant_solutions.set_global((((ant_index) * (d_n_objects)) + (step)), (selected_object));
				d_ant_available_objects.set_global((((ant_index) * (d_n_objects)) + (selected_object)), 0);
				for(int j = 0; ((j) < (d_n_constraints)); j++){
					d_free_space.set_global((((ant_index) * (d_n_constraints)) + (j)), (d_free_space.get_global((((ant_index) * (d_n_constraints)) + (j)))/* TODO: For multiple GPUs*/ - dimensions_values.get_global((((j) * (d_n_objects)) + (selected_object)))/* TODO: For multiple GPUs*/));
				}
				fitness = (object_values.get_global((selected_object))/* TODO: For multiple GPUs*/ + (fitness));
				}
				 else {
						d_ant_solutions.set_global((((ant_index) * (d_n_objects)) + (step)), -(1));
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
	
	
	
	
	
	
	
	int main(int argc, char** argv) {
		mkt::init();
		
		
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point complete_timer_start = std::chrono::high_resolution_clock::now();
		
		mkt::DArray<int> object_values(0, 6, 6, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> dimensions_values(0, 60, 60, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> constraints_max_values(0, 60, 60, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> dimension_values(0, 60, 60, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> constraint_max_values(0, 10, 10, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> d_ant_solutions(0, 6144, 6144, -1, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> d_best_solution(0, 6, 6, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> d_ant_available_objects(0, 6144, 6144, 1, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<double> d_pheromones(0, 36, 36, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<double> d_delta_phero(0, 36, 36, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<double> d_probabilities(0, 6144, 6144, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> d_free_space(0, 10240, 10240, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<double> d_eta(0, 6144, 6144, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<double> d_tau(0, 6144, 6144, 0.0, 1, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DArray<int> d_ant_fitness(0, 1024, 1024, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
		
		InitPheros_map_index_in_place_array_functor initPheros_map_index_in_place_array_functor{};
		InitFreeSpace_map_index_in_place_array_functor initFreeSpace_map_index_in_place_array_functor{constraint_max_values};
		Generate_solutions_map_index_in_place_array_functor generate_solutions_map_index_in_place_array_functor{d_ant_available_objects, object_values, d_pheromones, dimensions_values, d_free_space, d_eta, d_tau, d_probabilities, d_ant_solutions, constraint_max_values};
		Evaporate_map_index_in_place_array_functor evaporate_map_index_in_place_array_functor{};
		
		
				
		
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		int n_iterations = 5;
		double evaporation = 0.5;
		int n_objects = 0;
		int n_constraints = 0;
		double Q = 0.0;
		double best_fitness = 0.0;
		double mean_times = 0.0;
		int n_ants = 1024;
		mkt::map_index_in_place<double, InitPheros_map_index_in_place_array_functor>(d_pheromones, initPheros_map_index_in_place_array_functor);
		initFreeSpace_map_index_in_place_array_functor.n_ants = (n_ants);
		mkt::map_index_in_place<int, InitFreeSpace_map_index_in_place_array_functor>(d_free_space, initFreeSpace_map_index_in_place_array_functor);
		int iteration = 0;
		for(int ii = 0; ((ii) < (n_iterations)); ii++){
			generate_solutions_map_index_in_place_array_functor.d_n_objects = 6;generate_solutions_map_index_in_place_array_functor.d_n_constraints = 10;
			mkt::map_index_in_place<int, Generate_solutions_map_index_in_place_array_functor>(d_ant_fitness, generate_solutions_map_index_in_place_array_functor);
			for(int i = 0; ((i) < (n_ants)); i++){
				double ant_j_fitness = d_ant_fitness.get_global((i))/* TODO: For multiple GPUs*/;
				if(((ant_j_fitness) > (best_fitness))){
					best_fitness = (ant_j_fitness);
					for(int j = 0; ((j) < (n_objects)); j++){
						d_best_solution = d_ant_solutions.get_global((((i) * (n_objects)) + (j)))/* TODO: For multiple GPUs*/;
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
