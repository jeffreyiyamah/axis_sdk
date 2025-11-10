#include "axis/vo_handler.h"
#include <iostream>


void axis::VoHandler::feedVO() {
    health_ = SensorHealth::OFFLINE;
    std::cerr << "VO handler: Visual odometry integration not yet implemented.\n";
}