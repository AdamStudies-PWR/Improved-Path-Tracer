#pragma once

#include <stdint.h>
#include <string>

namespace tracer::utils
{

class InputParser
{
public:
    InputParser(int argumentCount, char* argumentList[]);

    bool isInputValid();
    std::string getScenePath();
    uint16_t getSamplingRate();

private:
    bool validateInput(int argumentCount, char* argumentList[]);
    bool validatePath(std::string path);
    bool validateSamples(std::string number);
    void printErrorMessage(std::string error);

    bool isValid_;
    std::string scenePath_;
    uint16_t sampleRate_;
};

}  // namespace tracer::utils
