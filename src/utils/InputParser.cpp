#include "utils/InputParser.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>

namespace tracer::utils
{

namespace
{
const uint8_t EXPECTED_ARGUMENT_COUNT_MIN = 1;
const uint8_t EXPECTED_ARGUMENT_COUNT_MAX = 3;
const uint8_t DEFAULT_DEPTH = 10;
const uint8_t MIN_DEPTH = 3;
const uint8_t MAX_DEPTH = 255;
const uint16_t DEFAULT_SAMPLES = 40;
const uint16_t MIN_SAMPLES = 4;
const uint16_t MAX_SAMPLES = 65535;
const std::string HELP_REQUEST = "--help";
const std::vector<std::string> SHORT_ALLOWED_ARGS = {"s", "d"};
const std::vector<std::string> LONG_ALLOWED_ARGS = {"samples", "depth"};

std::vector<std::string> split(std::string input, const std::string& separator)
{
    size_t pos = 0;
    std::vector<std::string> out = {};

    while ((pos = input.find(separator)) != std::string::npos)
    {
        out.push_back(input.substr(0, pos));
        input.erase(0, pos + separator.length());
    }
    out.push_back(input);

    return out;
}

std::string getSceneNameFromFile(std::string str)
{
    auto posSlash = str.rfind("/");
    if (posSlash != std::string::npos)
    {
        str.replace(str.begin(), str.begin() + posSlash + 1, "");
    }
    auto posDot = str.rfind(".");
    if (posDot != std::string::npos)
    {
        str.replace(str.begin() + posDot, str.end(), "");
    }

    return str;
}

}  // namespace

InputParser::InputParser(int argumentCount, char* argumentList[])
    : maxDepth_(DEFAULT_DEPTH)
    , sampleRate_(DEFAULT_SAMPLES)
    , sceneName_("")
{
    isValid_ = validateInput(argumentCount, argumentList);
}

bool InputParser::isInputValid() const { return isValid_; }
std::string InputParser::getScenePath() const { return scenePath_; }
uint8_t InputParser::getMaxDepth() const { return maxDepth_; }
uint16_t InputParser::getSamplingRate() const { return sampleRate_; }
std::string InputParser::getSceneName() const { return sceneName_; }

bool InputParser::validateInput(const int argumentCount, char* argumentList[])
{
    if (argumentCount < EXPECTED_ARGUMENT_COUNT_MIN || argumentCount > EXPECTED_ARGUMENT_COUNT_MAX)
    {
        std::stringstream errorMessage;
        errorMessage << "Got " << argumentCount << " arguments! Expected between " << (int)EXPECTED_ARGUMENT_COUNT_MIN
                     << " and " << (int)EXPECTED_ARGUMENT_COUNT_MAX << " arguments";
        printErrorMessage(errorMessage.str());
        return false;
    }

    switch (argumentCount)
    {
        case 1:
        {
            if (isHelpRequest(argumentList[1])) return false;
            if (not validatePath(argumentList[1])) return false;
        } break;
        case 2:
        case 3:
        {
            if (not validatePath(argumentList[argumentCount])) return false;
            if (not validateArguments(argumentList, argumentCount)) return false;
        } break;
        default:
        {
            printErrorMessage("Could not parse arguments provided.");
            return false;
        }
    }

    return true;
}

bool InputParser::isHelpRequest(const std::string& arg)
{
    if (arg != HELP_REQUEST) return false;
    printHelpMessage();
    return true;
}

bool InputParser::validatePath(const std::string& path)
{
    if (not std::filesystem::exists(path))
    {
        printErrorMessage("Path does not exist");
        return false;
    }
    if (not std::filesystem::is_regular_file(path))
    {
        printErrorMessage("Not a file");
        return false;
    }

    sceneName_ = getSceneNameFromFile(path);
    scenePath_ = path;
    return true;
}

bool InputParser::validateArguments(char* argumentList[], const int argumentCount)
{
    for (uint8_t i = 1; i < argumentCount; i++)
    {
        auto arg = std::string(argumentList[i]);
        const auto slashes = std::count(arg.begin(), arg.end(), '-');

        if (slashes != 1 && slashes != 2)
        {
            printErrorMessage("Arguments can have 1 or 2 (-)! Please check your input");
            return false;
        }

        arg.erase(std::remove(arg.begin(), arg.end(), '-'), arg.end());
        const auto marker_arg = split(arg, "=");

        if (marker_arg.size() != 2)
        {
            printErrorMessage("Cannot parse argument: " + arg);
            return false;
        }

        if (slashes == 1)
        {
            if (std::find(SHORT_ALLOWED_ARGS.begin(), SHORT_ALLOWED_ARGS.end(), marker_arg[0])
                == SHORT_ALLOWED_ARGS.end())
            {
                printErrorMessage("Unknown short argument: " + arg);
                return false;
            }
        }
        else
        {
            if (std::find(LONG_ALLOWED_ARGS.begin(), LONG_ALLOWED_ARGS.end(), marker_arg[0]) == LONG_ALLOWED_ARGS.end())
            {
                printErrorMessage("Unknown long argument: " + arg);
                return false;
            }
        }

        if (marker_arg[0] == SHORT_ALLOWED_ARGS[0] || marker_arg[0] == LONG_ALLOWED_ARGS[0])
        {
            if (not validateSamples(marker_arg[1])) return false;
        }

        if (marker_arg[0] == SHORT_ALLOWED_ARGS[1] || marker_arg[0] == LONG_ALLOWED_ARGS[1])
        {
            if (not validateDepth(marker_arg[1])) return false;
        }
    }

    return true;
}

bool InputParser::validateSamples(const std::string& number)
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
        printErrorMessage("Could not convert samples to number!");
        return false;
    }

    if (temp < MIN_SAMPLES || temp > MAX_SAMPLES)
    {
        printErrorMessage("Number of samples out of range!");
        return false;
    }

    sampleRate_ = (uint16_t)temp;
    return true;
}

bool InputParser::validateDepth(const std::string& number)
{
    int temp;
    try
    {
        temp = std::stoi(number);
    }
    catch (std::out_of_range const& ex)
    {
        printErrorMessage("Depth out of range!");
        return false;
    }
    catch (std::invalid_argument const& ex)
    {
        printErrorMessage("Could not convert depth to number!");
        return false;
    }

    if (temp < MIN_DEPTH || temp > MAX_DEPTH)
    {
        printErrorMessage("Depth out of range!");
        return false;
    }

    maxDepth_ = (uint8_t)temp;
    return true;
}

void InputParser::printErrorMessage(const std::string& error)
{
    std::cout << "Error parsing input!" << std::endl;
    std::cout << "Cause: " << error << std::endl;
    std::cout << "Usage:" << std::endl;
    printHelpMessage();
}

void InputParser::printHelpMessage()
{
    std::cout << "tracer [arguments] [path_to_scene]" << std::endl;
    std::cout << "[arguments] are [-s/--samples] or [-d/--depth]" << std::endl;
    std::cout << "\t [OPTIONAL] -s=number or --samples=number - Specifies number of samples per pixel. "
              << "It must be between " << MIN_SAMPLES << " and " << MAX_SAMPLES << std::endl;
    std::cout << "\t [OPTIONAL] -d=number or --depth=number - Specifies max number of reflections per ray. "
              << "It must be between " << (int)MIN_DEPTH << " and " << (int)MAX_DEPTH << std::endl;
    std::cout << "[path_to_scene] - Specifies path to json file with scene data. It is mandatory." << std::endl;
}

}  // namespace tracer::utils
