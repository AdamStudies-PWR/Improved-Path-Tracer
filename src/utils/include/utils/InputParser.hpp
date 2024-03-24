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
    std::string getSceneName() const;

private:
    bool isHelpRequest(const std::string& arg);
    bool validateInput(const int argumentCount, char* argumentList[]);
    bool validatePath(const std::string& path);
    bool validateArguments(char* argumentList[], const int argumentCount);
    bool validateSamples(const std::string& number);
    bool validateDepth(const std::string& number);
    void printErrorMessage(const std::string& error);
    void printHelpMessage();

    bool isValid_;
    std::string sceneName_;
    std::string scenePath_;
    uint16_t sampleRate_;
    uint8_t maxDepth_;
};

}  // namespace tracer::utils
