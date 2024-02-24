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
    uint8_t getMaxDepth() const;
    uint16_t getSamplingRate() const;

private:
    bool validateInput(const int argumentCount, char* argumentList[]);
    bool validatePath(const std::string& path);
    bool validateArguments(char* argumentList[], const int argumentCount);
    bool validateSamples(const std::string& number);
    bool validateDepth(const std::string& number);
    void printErrorMessage(const std::string& error);

    bool isValid_;
    std::string scenePath_;
    uint8_t maxDepth_;
    uint16_t sampleRate_;
};

}  // namespace tracer::utils
