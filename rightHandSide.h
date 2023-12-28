//#----------------------------------------------------------
//#
//# This file defines the boundary and initial conditions
//#
//#----------------------------------------------------------

#ifndef GLOBAL_PARA
#define GLOBAL_PARA
#include "./globalPara.h"
#endif
  

//# Declaration
  
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide() : Function<dim>() {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  };

//# Implementation

  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());

    const double time = this->get_time();

    double position_square_x = global_init_position_x0 + global_V_scan_x * time;
    double position_square_y = global_init_position_y0 + global_V_scan_y * time;

    if ( (p[0] > position_square_x - square_length)  && (p[0] < position_square_x + square_length)
        &&
         (p[1] > position_square_y - square_length)  && (p[1] < position_square_y + square_length)
        ){
            return 2;
        }
    else{
        return 0;
    }
  }