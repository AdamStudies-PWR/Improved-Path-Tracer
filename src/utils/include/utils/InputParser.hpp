#pragma once

#include <stdint.h>
#include <string>

namespace tracer::utils
{

class InputParser
{
public:
    InputParser(int argumentCount, char* argumentList[]);

    bool isInputValid() const;
    std::string getScenePath() const;
    uint16_t getSamplingRate() const;

private:
    bool validateInput(const int argumentCount, char* argumentList[]);
    bool validatePath(const std::string& path);
    bool validateSamples(const std::string& number);
    void printErrorMessage(const std::string& error);

    bool isValid_;
    std::string scenePath_;
    uint16_t sampleRate_;
};

}  // namespace tracer::utils
