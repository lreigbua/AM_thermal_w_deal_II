

// physics constants
double global_PI = 3.1415927;

// Heat acitvation area
double square_length = 0.1;

//Convection
double conv_heat_loss = -0.001;


// # Laser
double global_Pow_laser = 10;      // power [W]
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
double l_x = 0.25;
double l_y = 0.25;    
// recoating
int n_layers = 1;
double layer_thickness = l_y / n_layers;
double cooling_time = 0.1;

double global_V_scan_x = 5;   // scan speed, [m/s]

// # simulation time
double global_simulation_time_step = 1e-5;          // 10 [us]
double scanning_time = l_x / global_V_scan_x;
double layer_time = scanning_time + cooling_time;

double global_simulation_end_time = layer_time * n_layers; //    100 [um] / scan speed
