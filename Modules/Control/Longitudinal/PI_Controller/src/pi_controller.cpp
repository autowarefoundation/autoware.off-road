#include "pi_controller.hpp"

PI_Controller::PI_Controller(double K_p, double K_i, double K_d)
{
    std::cout << "PI controller init with Params \n"
              << "K_p\t" << K_p << "\n"
              << "K_i\t" << K_i << "\n"
              << "K_d\t" << K_d << "\n";

    integral_error = 0.0;
    prev_error = 0.0;
    this->K_p = K_p;
    this->K_i = K_i;
    this->K_d = K_d;
}

double PI_Controller::computeEffort(double current_speed_, double target_speed_)
{
    double error = target_speed_ - current_speed_;
    integral_error += error;
    double effort = K_p * error + K_i * integral_error + (error - prev_error) * K_d;
    prev_error = error;
    return effort;
}

extern "C" {
    void* create_pi_controller(double K_p, double K_i, double K_d) {
        return new PI_Controller(K_p, K_i, K_d);
    }

    void destroy_pi_controller(void* handle) {
        if (handle) {
            delete static_cast<PI_Controller*>(handle);
        }
    }

    double pi_compute_effort(void* handle, double current_speed, double target_speed) {
        if (!handle) return 0.0;
        PI_Controller* controller = static_cast<PI_Controller*>(handle);
        return controller->computeEffort(current_speed, target_speed);
    }
}