/* Code developed by Luis Reig modifying the code provided in Step-26 of the Deal.II tutorial (https://www.dealii.org/current/doxygen/deal.II/step_26.html). 
Includes modifications to activate elements in time and move the heat source in time.
*/

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
#include <cmath>
#include <algorithm>

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
    , time_step(global_simulation_time_step)
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

  // Deactivates all elements except the ones in the first layer
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

    data_out.add_data_vector(fe_degrees, "fe_degree", DataOut<dim>::type_cell_data);

    data_out.add_data_vector(solution, "Temperature", DataOut<dim>::type_dof_data);

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
    const double geometric_tolerance = 1e-10;
    const double layer_event_tolerance = 1e-10;
    const double layer_remainder = std::fmod(time, layer_time);
    const bool activating_layer =
      (time > layer_event_tolerance) &&
      (layer_remainder < layer_event_tolerance ||
       layer_time - layer_remainder < layer_event_tolerance);

    const unsigned int active_layers =
      std::max(1u,
               std::min(n_layers,
                        static_cast<unsigned int>(
                          std::floor((time + geometric_tolerance) / layer_time)) +
                          1));

    const double layer_top =
      std::min(l_y, static_cast<double>(active_layers) * layer_thickness);
    const double layer_bottom = std::max(0.0, layer_top - layer_thickness);

    bool mesh_changed = false;

    // Refine cells inside the newest layer and coarsen all other layers.
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        const double cell_center_height = cell->center()[1];

        if (cell->level() < max_grid_level &&
            cell_center_height >= layer_bottom - geometric_tolerance &&
            cell_center_height <= layer_top + geometric_tolerance)
          {
            cell->set_refine_flag();
            mesh_changed = true;
          }
        else if (cell->level() > min_grid_level &&
                 (cell_center_height < layer_bottom - geometric_tolerance ||
                  cell_center_height > layer_top + geometric_tolerance))
          {
            cell->set_coarsen_flag();
            mesh_changed = true;
          }
        else
          {
            cell->clear_refine_flag();
            cell->clear_coarsen_flag();
          }
      }

    if (!mesh_changed && !activating_layer)
      return;

    parallel::distributed::SolutionTransfer<dim, Vector<double>> solution_trans(
      dof_handler, true);

    Vector<double> previous_solution = solution;

    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    if (activating_layer)
      {
        activate_FEs(); // using set_future_FE_index()
      }

    triangulation.execute_coarsening_and_refinement();
    setup_system();

    if (activating_layer)
      {
        solution = 50.;
      }

    solution_trans.interpolate(solution);
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

    std::vector<unsigned int> reps =  {static_cast<unsigned int>(l_x/layer_thickness), n_layers};
    // GridGenerator::hyper_rectangle(triangulation, p1 , p2);
    
    GridGenerator::subdivided_hyper_rectangle(triangulation, reps, p1 , p2);

    triangulation.refine_global(initial_global_refinement);
  }

  template <int dim>
  void HeatEquation<dim>::run()
  {
    Create_Initial_Triangulation();
    
    deactivate_FEs();
    setup_system();

    const unsigned int min_grid_level = initial_global_refinement;
    const unsigned int max_grid_level =
      initial_global_refinement + geometry_refinement_cycles;
    const unsigned int refinement_passes_per_layer =
      std::max(1u, geometry_refinement_cycles);

    time = 0.0;
    for (unsigned int pass = 0; pass < refinement_passes_per_layer; ++pass)
      refine_mesh(min_grid_level, max_grid_level);
    unsigned int layers_refined = (n_layers == 0 ? 0 : 1);

    Vector<double> tmp;
    Vector<double> forcing_terms;

    timestep_number = 0;

    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());


    VectorTools::interpolate(dof_handler,
                             Functions::ConstantFunction<dim>(0.),
                             old_solution);
    solution = old_solution;

    
    output_results();

    // Then we start the main loop until the computed time exceeds our
    // end time. The first task is to build the right hand
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
        

        solve_time_step();

        output_results();

        const double layer_tolerance = 1e-10;
        const double remainder = std::fmod(time, layer_time);
        const bool new_layer_added =
          (time > layer_tolerance) &&
          (remainder < layer_tolerance ||
           layer_time - remainder < layer_tolerance);

        const unsigned int active_layers =
          std::max(1u,
                   std::min(n_layers,
                            static_cast<unsigned int>(
                              std::floor((time + layer_tolerance) /
                                         layer_time)) +
                              1));

        if (new_layer_added && active_layers > layers_refined)
          {
            for (unsigned int pass = 0; pass < refinement_passes_per_layer;
                 ++pass)
              refine_mesh(min_grid_level, max_grid_level);
            tmp.reinit(solution.size());
            forcing_terms.reinit(solution.size());
            layers_refined = active_layers;
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
