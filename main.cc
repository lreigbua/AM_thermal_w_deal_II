/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level output_directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */


// The program starts with the usual include files, all of which you should
// have seen before by now:
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>
#include <filesystem>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/refinement.h>

//try to use parallel solution transfer
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/numerics/point_value_history.h>

#include <deal.II/base/hdf5.h>

#include <vector>

// Then the usual placing of all content of this program into a namespace and
// the importation of the deal.II namespace into the one we will work in:
namespace Step26

{
  using namespace dealii;

  // Including the header file for the right hand side and boundary values
  #ifndef GLOBAL_PARA
  #define GLOBAL_PARA
  #include "./globalPara.h"
  // #include "./boundaryInit.h"
  #include "./rightHandSide.h"
  #endif


  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation();
    void run();

  private:
    void setup_system();
    void solve_time_step();
    void output_results();
    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);
    void Write_History_Output();
    
    // methods developed by me:
    // void set_active_FEs();
    void activate_FEs();
    void deactivate_FEs();

    void Create_Initial_Triangulation();

    MPI_Comm mpi_communicator;

    // Triangulation<dim> triangulation;
    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;



    double       time;
    double       time_step;
    unsigned int timestep_number;


    const double theta;

    std::string output_directory;

    // attributes developed by me:
    hp::FECollection<dim>    fe_collection;
    hp::QCollection<dim>     quadrature_collection;
    hp::QCollection<dim - 1> face_quadrature_collection;

    std::vector<double> History_Output_Value;
    std::vector<double> History_Output_Time;

    // HDF5::DataSet History_Output;

  };

  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };



  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                    const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }


  template <int dim>
  HeatEquation<dim>::HeatEquation()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(MPI_COMM_WORLD)
    , fe(1)
    , dof_handler(triangulation)
    , time_step(2. / 500)
    , theta(0.5)
    , output_directory("output")
  {
    fe_collection.push_back(FE_Q<dim>(1));
    fe_collection.push_back(FE_Nothing<dim>(1, true));

    quadrature_collection.push_back(QGauss<dim>(2));
    quadrature_collection.push_back(QGauss<dim>(2));

    face_quadrature_collection.push_back(QGauss<dim - 1>(2));
    face_quadrature_collection.push_back(QGauss<dim - 1>(2));    
  }

  // Note that we do not take the hanging node constraints into account when
  // assembling the matrices (both functions have an AffineConstraints argument
  // that defaults to an empty object). This is because we are going to
  // condense the constraints in run() after combining the matrices for the
  // current time-step.
  template <int dim>
  void HeatEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe_collection);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      quadrature_collection,
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         quadrature_collection,
                                         laplace_matrix);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  // Method to specify the active and inactive FEs from the triangulation, which change every time powder is added.
  template <int dim>
  void HeatEquation<dim>::deactivate_FEs()
  {

    for (auto &cell: dof_handler.active_cell_iterators())
    {
      for (const auto v : cell->vertex_indices())
      {
          Point<dim> Vertex_Point = cell->vertex(v);
          if (Vertex_Point[1] > layer_thickness + 1e-6){
            cell->set_active_fe_index(1);  // index 1 is for FE_Nothing elements, which are elements with 0 degrees of freedom, since it is the second element of the fe_collection array.
            break;
          }else{
            cell->set_active_fe_index(0);
          }
      }
    }
    dof_handler.distribute_dofs(fe_collection);
  }

  // Method to activate cells
  template <int dim>
  void HeatEquation<dim>::activate_FEs()
  {

    for (auto &cell: dof_handler.active_cell_iterators())
    {
      for (const auto v : cell->vertex_indices())
      {
          Point<dim> Vertex_Point = cell->vertex(v);
          if (Vertex_Point[1] <= layer_thickness + static_cast<int>(std::roundl(time / layer_time)) * layer_thickness - 1e-9){
            cell->set_future_fe_index(0);  // index 1 is for FE_Nothing elements, which are elements with 0 degrees of freedom, since it is the second element of the fe_collection array.
            break;
          }
      }
    }

  }

  template <int dim>
  void HeatEquation<dim>::solve_time_step()
  {
    SolverControl   solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step() << " CG iterations."
              << std::endl;
  }



  template <int dim>
  void HeatEquation<dim>::output_results()
  {
    
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);

    //Output de degrees of freedom in every element:
    Vector<float> fe_degrees(triangulation.n_active_cells());
      for (const auto &cell : dof_handler.active_cell_iterators())
        fe_degrees(cell->active_cell_index()) =
          fe_collection[cell->active_fe_index()].degree;

    data_out.add_data_vector(fe_degrees, "fe_degree");

    data_out.add_data_vector(solution, "Temperature");

    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
    
    

    // create output output_directory if it does not exist:
    if (!std::filesystem::exists(output_directory)) (std::filesystem::create_directory(output_directory));

    //Save to vtk file:
    const std::string filename =
      output_directory + "/solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
    std::ofstream output(filename);

    data_out.write_vtk(output);

    //save to hdf5 file:
    DataOutBase::DataOutFilterFlags flags(true, true);
    DataOutBase::DataOutFilter data_filter(flags);
    data_out.write_filtered_data(data_filter);

    const std::string filename_hdf5 =
      output_directory + "/solution-" + Utilities::int_to_string(timestep_number, 3) + ".hdf5";
    std::ofstream output_hdf5(filename_hdf5);

    data_out.write_hdf5_parallel(data_filter, filename_hdf5, MPI_COMM_WORLD);

    // add time to hdf5 file:
    HDF5::File data_file(filename_hdf5, HDF5::File::FileAccessMode::open, MPI_COMM_WORLD);
    Vector<double> time_f5(1);
    time_f5[0] = time;
    data_file.write_dataset("time", time_f5);

    Vector<double> point_temperature(1);
    point_temperature[0] = VectorTools::point_value(dof_handler, solution,
                                              Point<2>(0.1, 0.1));
    data_file.write_dataset("Point_Temperature", point_temperature);

    History_Output_Time.push_back(time);
    History_Output_Value.push_back(point_temperature[0]);
        
  }

  template <int dim>
  void HeatEquation<dim>::Write_History_Output()
  { 
    std::vector<hsize_t> n_time_steps = {History_Output_Value.size()};
    std::string filename = output_directory + "/History_Output.h5";
    HDF5::File data_file(filename, HDF5::File::FileAccessMode::create);
    auto group = data_file.create_group("history_output");
    auto time = group.create_dataset<double>("time", n_time_steps);
    auto temperature = group.create_dataset<double>("temperature", n_time_steps);
    time.write(History_Output_Time);
    temperature.write(History_Output_Value);
  }


  template <int dim>
  void HeatEquation<dim>::refine_mesh(const unsigned int min_grid_level,
                                      const unsigned int max_grid_level)
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      face_quadrature_collection,
      std::map<types::boundary_id, const Function<dim> *>(),
      solution,
      estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.55,
                                                      0.45);

    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();


    // SolutionTransfer<dim> solution_trans(dof_handler);

    parallel::distributed::SolutionTransfer< dim, Vector<double>> solution_trans(dof_handler, true);

    Vector<double> previous_solution;
    previous_solution = solution;
    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    if (time > 1e-10 && std::fmod(time, layer_time) < 1e-10){
      activate_FEs(); // using set_future_FE_index()
    }

    triangulation.execute_coarsening_and_refinement();
    setup_system();

    if (time > 1e-10 && std::fmod(time, layer_time) < 1e-10){
      solution = 50.;
    }

    solution_trans.interpolate(solution);
    // solution_trans.interpolate(previous_solution, solution); this is for not paralel distributed triangulations
    constraints.distribute(solution);
  }

  template <int dim>
  void HeatEquation<dim>::Create_Initial_Triangulation()
  {
    Point<dim> p1;
    Point<dim> p2;

    std::cout << "dim = " << dim << std::endl;

    if (dim == 2){
      p1 = {0, 0};
      p2 = {l_x, l_y};
    } else if (dim == 3){
      p1 = {0, 0, 0};
      p2 = {l_x, l_y, l_y};
    }

    GridGenerator::hyper_rectangle(triangulation, p1 , p2);
    triangulation.refine_global(initial_global_refinement);
  }

  template <int dim>
  void HeatEquation<dim>::run()
  {
    const unsigned int n_adaptive_pre_refinement_steps = 3;

    Create_Initial_Triangulation();
    
    deactivate_FEs();
    setup_system();

    unsigned int pre_refinement_step = 0;

    Vector<double> tmp;
    Vector<double> forcing_terms;

  start_time_iteration:

    time            = 0.0;
    timestep_number = 0;

    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());


    VectorTools::interpolate(dof_handler,
                             Functions::ConstantFunction<dim>(0.),
                             old_solution);
    solution = old_solution;

    
    output_results();

    // Then we start the main loop until the computed time exceeds our
    // end time of 0.5. The first task is to build the right hand
    // side of the linear system we need to solve in each time step.
    // Recall that it contains the term $MU^{n-1}-(1-\theta)k_n AU^{n-1}$.
    // We put these terms into the variable system_rhs, with the
    // help of a temporary vector:
    while (time <= global_simulation_end_time)
      {
        
        time += time_step;
        ++timestep_number;

        

        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        mass_matrix.vmult(system_rhs, old_solution);

        laplace_matrix.vmult(tmp, old_solution);
        system_rhs.add(-(1 - theta) * time_step, tmp);

        // The second piece is to compute the contributions of the source
        // terms. This corresponds to the term $k_n
        // \left[ (1-\theta)F^{n-1} + \theta F^n \right]$. The following
        // code calls VectorTools::create_right_hand_side to compute the
        // vectors $F$, where we set the time of the right hand side
        // (source) function before we evaluate it. The result of this
        // all ends up in the forcing_terms variable:
        RightHandSide<dim> rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler,
                                            quadrature_collection,
                                            rhs_function,
                                            tmp);
        
        // add convection to rhs
        Vector<double> tmp2(tmp.size());
        VectorTools::create_boundary_right_hand_side(
          dof_handler,
          face_quadrature_collection,
          Functions::ConstantFunction<dim>(conv_heat_loss),
          tmp2);
        
        tmp += tmp2;

        forcing_terms = tmp;
        forcing_terms *= time_step * theta;

        rhs_function.set_time(time - time_step);
        VectorTools::create_right_hand_side(dof_handler,
                                            quadrature_collection,
                                            rhs_function,
                                            tmp);

        // add convection to rhs
        VectorTools::create_boundary_right_hand_side(
          dof_handler,
          face_quadrature_collection,
          Functions::ConstantFunction<dim>(conv_heat_loss),
          tmp2);
        
        tmp += tmp2;

        forcing_terms.add(time_step * (1 - theta), tmp);

        // Next, we add the forcing terms to the ones that
        // come from the time stepping, and also build the matrix
        // $M+k_n\theta A$ that we have to invert in each time step.
        // The final piece of these operations is to eliminate
        // hanging node constrained degrees of freedom from the
        // linear system:
        system_rhs += forcing_terms;

        system_matrix.copy_from(mass_matrix);
        system_matrix.add(theta * time_step, laplace_matrix);

        constraints.condense(system_matrix, system_rhs);

        // There is one more operation we need to do before we
        // can solve it: boundary values. To this end, we create
        // a boundary value object, set the proper time to the one
        // of the current time step, and evaluate it as we have
        // done many times before. The result is used to also
        // set the correct boundary values in the linear system:
        {
          // BoundaryValues<dim> boundary_values_function;
          // boundary_values_function.set_time(time);

          // std::map<types::global_dof_index, double> boundary_values;
          // VectorTools::interpolate_boundary_values(dof_handler,
          //                                          0,
          //                                          boundary_values_function,
          //                                          boundary_values);

          // MatrixTools::apply_boundary_values(boundary_values,
          //                                    system_matrix,
          //                                    solution,
          //                                    system_rhs);
        }

        // With this out of the way, all we have to do is solve the
        // system, generate graphical data, and...
        solve_time_step();

        output_results();

        // ...take care of mesh refinement. Here, what we want to do is
        // (i) refine the requested number of times at the very beginning
        // of the solution procedure, after which we jump to the top to
        // restart the time iteration, (ii) refine every fifth time
        // step after that.
        //
        // The time loop and, indeed, the main part of the program ends
        // with starting into the next time step by setting old_solution
        // to the solution we have just computed.
        if ((timestep_number == 1) &&
            (pre_refinement_step < n_adaptive_pre_refinement_steps))
          {
            refine_mesh(initial_global_refinement,
                        initial_global_refinement +
                          n_adaptive_pre_refinement_steps);
            ++pre_refinement_step;

            tmp.reinit(solution.size());
            forcing_terms.reinit(solution.size());

            std::cout << std::endl;

            goto start_time_iteration;
          }
        else if ((timestep_number > 0) && (timestep_number % 5 == 0))
          {
            refine_mesh(initial_global_refinement,
                        initial_global_refinement +
                          n_adaptive_pre_refinement_steps);
            tmp.reinit(solution.size());
            forcing_terms.reinit(solution.size());
          }

        old_solution = solution;
      }

    Write_History_Output();
  }
} // namespace Step26


int main(int argc, char *argv[])
{
  try
    {
      using namespace Step26;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      HeatEquation<2> heat_equation_solver;
      heat_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
