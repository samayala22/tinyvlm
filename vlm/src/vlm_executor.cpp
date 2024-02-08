#include "vlm_executor.hpp"

using namespace vlm;

std::unique_ptr<tf::Executor> Executor::_instance = nullptr;
