// physics constants
double global_PI = 3.1415927;

// Heat acitvation area
double square_length = 0.075;

//Convection
double conv_heat_loss = -100.0;

// # Laser
double global_Pow_laser = 100000;      // power [W]
double global_spotsize_at_e_2 = 20e-6;  // laser spot size at e^(-2) [m]
double global_c_laser = global_spotsize_at_e_2 / 4.0;   // C parameter in Gaussian func, [m]
double global_c_hwhm = global_c_laser * 2.35482 / 2;    // HWHM, [m]

double global_init_position_x0 = 0;    // initial spot center position
double global_init_position_y0 = 1;

// # material
// thin film
double global_rho_Tio2 = 4200;      // mass density, [kg/m^3]
double global_C_Tio2 = 690;         // heat capacity, [J/kg/K]
double global_k_Tio2 = 4.8;         // thermal conductivity, [W/m/K]

// substrate
double global_rho_glass = 2200;
double global_C_glass = 700;
double global_k_glass = 1.8;

double global_film_thickness = 400e-9;  // film thickness, [m]


// # about the MESH
#define BOUNDARY_NUM 11


// Domain
double l_x = 2.;
double l_y = 1.25;

// refinement
int initial_global_refinement = 0;
int min_refinement_level = 0;
int max_refinement_level = 2;
int refinement_passes = max_refinement_level - min_refinement_level;
double lenght_refine_box = l_x / 2;
double padding_before_refinement = l_x / 8;

// recoating
double layer_thickness = 0.125;
unsigned int n_layers = static_cast<unsigned int>(l_y/layer_thickness);
double cooling_time = 0.2;

double global_V_scan_x = 20;   // scan speed, [m/s]

// # simulation time
double global_simulation_time_step = 0.002;          // 10 [us]
double scanning_time = l_x / global_V_scan_x;
double layer_time = scanning_time + cooling_time;
double global_simulation_end_time = layer_time * n_layers; //    100 [um] / scan speed

// tolerances
double global_tolerance = 1e-10;
