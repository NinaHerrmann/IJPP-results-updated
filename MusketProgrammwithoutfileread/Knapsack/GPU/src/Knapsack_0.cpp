	
	#include <omp.h>
	#include <openacc.h>
	#include <stdlib.h>
	#include <math.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	#include <accelmath.h>
	//#include <cuda.h>
	//#include <openacc_curand.h>
	
	#include "../include/musket.hpp"
	#include "../include/Knapsack_0.hpp"
	
	
	
			
	const int global_object_values = 6;
	const int global_n_constraints = 10;
	const double global_Q = 0.0;
	
	

	
	struct InitPheros_map_index_in_place_array_functor{
		
		InitPheros_map_index_in_place_array_functor(){
		}
		
		~InitPheros_map_index_in_place_array_functor() {}
		
		auto operator()(int i, double y){
			return 1.2;
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct InitFreeSpace_map_index_in_place_array_functor{
		
		InitFreeSpace_map_index_in_place_array_functor(const mkt::DArray<int>& _constraint_max_values) : constraint_max_values(_constraint_max_values){
		}
		
		~InitFreeSpace_map_index_in_place_array_functor() {}
		
		auto operator()(int i, int y){
			int j = (static_cast<int>((i)) % (n_ants));
			return // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
		}
	
		void init(int gpu){
			constraint_max_values.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		int n_ants;
		
		mkt::DeviceArray<int> constraint_max_values;
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Generate_solutions_map_index_in_place_array_functor{
		
		Generate_solutions_map_index_in_place_array_functor(const mkt::DArray<int>& _d_ant_available_objects, const mkt::DArray<int>& _object_values, const mkt::DArray<double>& _d_pheromones, const mkt::DArray<int>& _dimensions_values, const mkt::DArray<int>& _d_free_space, const mkt::DArray<double>& _d_eta, const mkt::DArray<double>& _d_tau, const mkt::DArray<double>& _d_probabilities, const mkt::DArray<int>& _d_ant_solutions, const mkt::DArray<int>& _constraint_max_values) : d_ant_available_objects(_d_ant_available_objects), object_values(_object_values), d_pheromones(_d_pheromones), dimensions_values(_dimensions_values), d_free_space(_d_free_space), d_eta(_d_eta), d_tau(_d_tau), d_probabilities(_d_probabilities), d_ant_solutions(_d_ant_solutions), constraint_max_values(_constraint_max_values){
		}
		
		~Generate_solutions_map_index_in_place_array_functor() {}
		
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
					
					if((// TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed == 1)){
					value_object_j = // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
					pheromone_to_object_j = // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
					average_tightness_object_j = 0.0;
					is_too_big = false;
					for(int ii = 0; ((ii) < (d_n_constraints)); ii++){
						size_i_object_j = // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
						free_space_i = // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
						
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
					eta = pow(((value_object_j) / (average_tightness_object_j)), 1);
					tau = pow((pheromone_to_object_j), 1);
					eta_tau_sum = (((eta) * (tau)) + (eta_tau_sum));
					d_eta[(((ant_index) * (d_n_objects)) + (object_j))] = (eta);
					d_tau[(((ant_index) * (d_n_objects)) + (object_j))] = (tau);
					is_possible = true;
					}
					 else {
							d_eta[(((ant_index) * (d_n_objects)) + (object_j))] = 0.0;
							d_tau[(((ant_index) * (d_n_objects)) + (object_j))] = 0.0;
						}
					}
					 else {
							d_eta[(((ant_index) * (d_n_objects)) + (object_j))] = 0.0;
							d_tau[(((ant_index) * (d_n_objects)) + (object_j))] = 0.0;
						}
				}
				
				if((is_possible)){
				for(int object_j = 0; ((object_j) < (d_n_objects)); object_j++){
					d_probabilities[(((ant_index) * (d_n_objects)) + (object_j))] = ((// TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed * // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed) / (eta_tau_sum));
				}
				double random = 0.0;
				select_index = 0;
				int selected_object = 0;
				double sum = 0.0;
				double prob = 0.0;
				for(int whilei = 0; ((whilei) > 0); whilei++){
					prob = // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
					
					if(((prob) > 0.0)){
					sum = ((sum) + (prob));
					selected_object = (select_index);
					}
					select_index = ((select_index) + 1);
					
					if(((sum) <= (random))){
					
					if(((select_index) < (d_n_objects))){
					whilei = -(1);
					}
					}
				}
				d_ant_solutions[(((ant_index) * (d_n_objects)) + (step))] = (selected_object);
				d_ant_available_objects[(((ant_index) * (d_n_objects)) + (selected_object))] = 0;
				for(int j = 0; ((j) < (d_n_constraints)); j++){
					d_free_space[(((ant_index) * (d_n_constraints)) + (j))] = (// TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed - // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed);
				}
				fitness = (// TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed + (fitness));
				}
				 else {
						d_ant_solutions[(((ant_index) * (d_n_objects)) + (step))] = -(1);
					}
			}
			for(int j = 0; ((j) < (d_n_constraints)); j++){
				d_free_space[(((ant_index) * (d_n_constraints)) + (j))] = // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
			}
			for(int j = 0; ((j) < (d_n_objects)); j++){
				d_ant_available_objects[(((ant_index) * (d_n_objects)) + (j))] = 1;
			}
			return (fitness);
		}
	
		void init(int gpu){
			d_ant_available_objects.init(gpu);
			object_values.init(gpu);
			d_pheromones.init(gpu);
			dimensions_values.init(gpu);
			d_free_space.init(gpu);
			d_eta.init(gpu);
			d_tau.init(gpu);
			d_probabilities.init(gpu);
			d_ant_solutions.init(gpu);
			constraint_max_values.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
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
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Evaporate_map_index_in_place_array_functor{
		
		Evaporate_map_index_in_place_array_functor(){
		}
		
		~Evaporate_map_index_in_place_array_functor() {}
		
		auto operator()(int i, double value){
			return ((value) * (evaporation));
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		double evaporation;
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Pheromone_deposit_map_index_in_place_array_functor{
		
		Pheromone_deposit_map_index_in_place_array_functor(const mkt::DArray<int>& _d_ant_solutions, const mkt::DArray<int>& _object_values, const mkt::DArray<double>& _d_pheromones) : d_ant_solutions(_d_ant_solutions), object_values(_object_values), d_pheromones(_d_pheromones){
		}
		
		~Pheromone_deposit_map_index_in_place_array_functor() {}
		
		auto operator()(int iindex, int pherovalue){
			int ant_index = ((iindex) % (n_objects));
			int i = ((iindex) % (n_ants));
			int object_i = 0;
			double delta_phero = 0.0;
			int value = 0;
			object_i = // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
			
			if(((object_i) != -(1))){
			value = // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
			delta_phero = (static_cast<double>((global_Q)) * (value));
			d_pheromones[static_cast<int>((((i) * (n_objects)) + (object_i)))] = (delta_phero);
			}
			return -(1);
		}
	
		void init(int gpu){
			d_ant_solutions.init(gpu);
			object_values.init(gpu);
			d_pheromones.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		int n_objects;
		int n_ants;
		
		mkt::DeviceArray<int> d_ant_solutions;
		mkt::DeviceArray<int> object_values;
		mkt::DeviceArray<double> d_pheromones;
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
		
		mkt::wait_all();
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
		Pheromone_deposit_map_index_in_place_array_functor pheromone_deposit_map_index_in_place_array_functor{d_ant_solutions, object_values, d_pheromones};
		
		
				
		
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
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
				double ant_j_fitness = // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
				if(((ant_j_fitness) > (best_fitness))){
					best_fitness = (ant_j_fitness);
					for(int j = 0; ((j) < (n_objects)); j++){
						d_best_solution = // TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed;
					}
				}
			}
			evaporate_map_index_in_place_array_functor.evaporation = (evaporation);
			mkt::map_index_in_place<double, Evaporate_map_index_in_place_array_functor>(d_pheromones, evaporate_map_index_in_place_array_functor);
			pheromone_deposit_map_index_in_place_array_functor.n_objects = (n_objects);pheromone_deposit_map_index_in_place_array_functor.n_ants = (n_ants);
			mkt::map_index_in_place<int, Pheromone_deposit_map_index_in_place_array_functor>(d_ant_solutions, pheromone_deposit_map_index_in_place_array_functor);
		}
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		
		mkt::wait_all();
		std::chrono::high_resolution_clock::time_point complete_timer_end = std::chrono::high_resolution_clock::now();
		double complete_seconds = std::chrono::duration<double>(complete_timer_end - complete_timer_start).count();
		printf("Complete execution time: %.5fs\n", complete_seconds);
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", 1);
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
