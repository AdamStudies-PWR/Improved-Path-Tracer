#include "utils/InputParser.hpp"

#include <iostream>
#include <sstream>

namespace tracer::utils
{

namespace
{
const uint8_t EXPECTED_ARGUMENT_COUNT = 2;
const uint16_t MIN_SAMPLES = 4;
const u_int16_t MAX_SAMPLES = 65535;
}

InputParser::InputParser(int argumentCount, char* argumentList[])
{
    isValid_ = validateInput(argumentCount, argumentList);
}

bool InputParser::isInputValid() { return isValid_; }
std::string InputParser::getScenePath() { return scenePath_; }
uint16_t InputParser::getSamplingRate() { return sampleRate_; }

bool InputParser::validateInput(int argumentCount, char* argumentList[])
{
    if (argumentCount != EXPECTED_ARGUMENT_COUNT)
    {
        std::stringstream errorMessage;
        errorMessage << "Got " << argumentCount << " arguments! Expected number: " << (int)EXPECTED_ARGUMENT_COUNT;
        printErrorMessage(errorMessage.str());
        return false;
    }

    if (not validatePath(argumentList[1])) return false;
    if (not validateSamples(argumentList[2])) return false;
    return true;
}

bool InputParser::validatePath(std::string path)
{
    // TODO: check if provided path exist and we can read from it
    scenePath_ = path;
    return true;
}

bool InputParser::validateSamples(std::string number)
{
    int temp;
    try
    {
        temp = std::stoi(number);
    }
    catch (std::out_of_range const& ex)
    {
        printErrorMessage("Number of samples out of range!");
        return false;
    }
    catch (std::invalid_argument const& ex)
    {
        printErrorMessage("Could not convert second argument to number!");
        return false;
    }

    if (temp < MIN_SAMPLES || temp > MAX_SAMPLES)
    {
        printErrorMessage("Number of samples out of range!");
        return false;
    }

    sampleRate_ = (uint16_t)temp / 4;
    return true;
}

void InputParser::printErrorMessage(std::string error)
{
    std::cout << "Error parsing input!" << std::endl;
    std::cout << "Cause: " << error << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "tracer [1] [2]" << std::endl;
    std::cout << "[1] - path to file containing scene data" << std::endl;
    std::cout << "[2] - Number of samples for each traced path. Must be a number between " << MIN_SAMPLES << "and "
              << MAX_SAMPLES << std::endl;
}

}  // namespace tracer::utils
