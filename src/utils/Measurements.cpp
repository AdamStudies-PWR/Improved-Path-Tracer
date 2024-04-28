#include "utils/Measurements.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

namespace tracer::utils
{

namespace
{
using namespace std::chrono;
using namespace containers;

const uint32_t HOUR_RATIO = 3600000;
const uint32_t MINUTES_RATIO = 60000;
const uint32_t SECONDS_RATIO = 1000;
const std::string BENCHMARK_FILE = "benchmark.txt";

std::string convertTimeUnitToString(uint32_t number)
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

void saveBenchmark(const std::string& id, const std::string time)
{
    std::fstream file;
    file.open(BENCHMARK_FILE, std::fstream::in | std::fstream::out | std::fstream::app);

    if (not file)
    {
        file.open(BENCHMARK_FILE,  std::fstream::in | std::fstream::out | std::fstream::trunc);
    }

    file << id << ";" << time << ";";
    file.close();
}
}  // namespace

const std::vector<Vec3> measure(const std::string& id, std::function<std::vector<Vec3>()> testable)
{
    std::cout << "Begining render..." << std::endl;
    auto start = high_resolution_clock::now();
    const auto image = testable();
    auto stop = high_resolution_clock::now();
    std::cout << " - Done" << std::endl;
    const auto time = getTimeString(duration_cast<milliseconds>(stop - start).count());
    std::cout << "Render took: " << time << std::endl;;
    saveBenchmark(id, time);

    return image;
}

}  // namespace tracer::utils
