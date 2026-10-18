#ifndef LLAMA_DECISION_MULTIMODAL_H
#define LLAMA_DECISION_MULTIMODAL_H

#if __has_include(<llama.h>) && __has_include(<mtmd.h>)

#include "./LLamaDecision.h"
#include <mtmd.h>
#include <variant>

#include "../Utils/Image2d.h"


struct DecisionAudio
{
    int sampleRate = 0;
    std::vector<float> samples; // Mono PCM F32 at GetAudioSampleRate().
};

using DecisionMedia = std::variant<Image2d<uint8_t>, DecisionAudio>;

// Stateless multimodal decisions, using the same projector API as the chat runner.
// Calls on the same instance must be serialized by the caller.
class LLamaDecisionMultiModal : public LLamaDecision
{
public:
    using LLamaDecision::Run;

    LLamaDecisionMultiModal(const std::string& modelPath, const std::string& projectorPath,
        const Settings& settings);
    ~LLamaDecisionMultiModal() override;

    void InitModel() override;
    void Relase() override;
    bool SupportsVision() const;
    bool SupportsAudio() const;
    int GetAudioSampleRate() const;
    
    std::unordered_map<std::string, DecisionAnswer> Run(const std::string& state,
        const std::vector<Question>& questions, const std::vector<DecisionMedia>& media);
    std::unordered_map<std::string, DecisionAnswer> Run(const std::vector<Question>& questions,
        const std::vector<DecisionMedia>& media);

protected:
    bool ProcessPrompt(DecisionPrompt& prompt) override;

private:
    
    std::string projectorPath;
    mtmd::context_ptr ctx_vision;

    bool LoadMediaModel();
    bool TokenizeMedia(const std::string& prompt, const std::vector<const mtmd_bitmap*>& bitmaps,
        DecisionPrompt& task) const;
};

#endif
#endif
