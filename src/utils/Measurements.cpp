#include "utils/Measurements.hpp"

#include <chrono>
#include <iostream>
#include <string>

namespace tracer::utils
{

namespace
{
using namespace std::chrono;

const uint32_t HOUR_RATIO = 3600000;
const uint32_t MINUTES_RATIO = 60000;
const uint32_t SECONDS_RATIO = 1000;
}

std::string convertTimeUnitToString(uint16_t number)
{
    return (number == 0) ? "00" : (number < 10) ? "0" + std::to_string(number) : std::to_string(number);
}

std::string getTimeString(uint64_t milliseconds)
{
    auto hours = milliseconds / HOUR_RATIO;
    const std::string hourString = convertTimeUnitToString(hours);
    milliseconds = milliseconds - HOUR_RATIO * hours;

    auto minutes = milliseconds / MINUTES_RATIO;
    const std::string minutesString = convertTimeUnitToString(minutes);
    milliseconds = milliseconds - MINUTES_RATIO * minutes;

    auto seconds = milliseconds / SECONDS_RATIO;
    const std::string secondsString = convertTimeUnitToString(seconds);
    milliseconds = milliseconds - SECONDS_RATIO * seconds;

    return hourString + ":" + minutesString + ":" + secondsString + "." + std::to_string(milliseconds);
}

containers::Vec* measure(std::function<containers::Vec*()> testable)
{
    std::cout << "Begining render..." << std::endl;
    auto start = high_resolution_clock::now();
    auto* image = testable();
    auto stop = high_resolution_clock::now();
    std::cout << " - Done" << std::endl;
    std::cout << "Render took: " << getTimeString(duration_cast<milliseconds>(stop - start).count()) << std::endl;;

    return image;
}

}  // namespace tracer::utils
