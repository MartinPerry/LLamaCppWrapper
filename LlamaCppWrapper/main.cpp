
#include <string>
#include <iostream>

#include "./LLamaWrapper.h"
#include "./LLamaChatRunner.h"
#include "./LLamaChat.h"
#include "./LLamaEmbedding.h"
#include "./LLamaChatMultiModal.h"
#include "./LLamaChatRunnerMultiModal.h"
#include "./LLamaUtils.h"

#include "./Decisions/LLamaDecision.h"
#include "./Decisions/LLamaDecisionMultiModal.h"


#include "./Utils/Logger.h"
#include "./Utils/Image2d.h"

//========================================================================================================

void testLlamaEmbedding()
{
	int ctxSize = 512;

	LLamaWrapper::Settings sets;
	sets.ctxSize = ctxSize;
	//sets.logicalBatchSize = ctxSize;
	//sets.physicalBatchSize = ctxSize;
	sets.gpuLayers = 999;
	sets.logEnabled = false;
	sets.extractEmbeddings = true;
	sets.onContextSetup = [](llama_context_params& ctx) {
		ctx.pooling_type = llama_pooling_type::LLAMA_POOLING_TYPE_MEAN;
		};

	std::shared_ptr<LLamaEmbedding> embed = std::make_shared<LLamaEmbedding>("E://_models_//llm_gguf//Qwen3-Embedding-4B-Q6_K.gguf", sets);

	embed->EnableCutOffWhenExceedBatch(true);

	//for Qwen3-Embedding-4B:
	//Each query must come with a one-sentence instruction that describes the task
	//No need to add instruction for retrieval documents

	auto res0 = embed->Run(std::string_view("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?"));
	auto res1 = embed->Run(std::string_view("The capital of China is Beijing."));

	printf("Finished");
}

//========================================================================================================

void testLlamaParallel()
{
	printf("====================\n");

	LLamaWrapper::Settings setsAddr;
	setsAddr.ctxSize = 4096;
	//setsAddr.logicalBatchSize = setsAddr.ctxSize;
	//setsAddr.physicalBatchSize = setsAddr.ctxSize;
	setsAddr.gpuLayers = 999;
	setsAddr.logEnabled = false;
	setsAddr.n_parallel = 3;

	//auto chat = std::make_shared<LLamaChat>("E://_models_//llm_gguf//multilingual-e5-large-instruct-f16.gguf", setsAddr);
	auto chatRunner = std::make_shared<LLamaChatRunner>("E://_models_//llm_gguf//deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf", setsAddr);
	chatRunner->InitModel();

	auto chat0 = chatRunner->GetChat(0);
	auto chat1 = chatRunner->GetChat(1);
	auto chat2 = chatRunner->GetChat(2);

	chat0->AddMessage("Is this url for page with webcam: https://chaleteliane.ch/webcam/");
	chat1->AddMessage("Is this url for page with webcam: https://meteo-oberwallis.ch/bettmeralp/alpe/");
	chat2->AddMessage("Is this url for page with webcam: https://chaletlucca.com/prodotto/camera-1/");
	chatRunner->Run();
	auto respon0 = chat0->GetLastMessage();
	auto respon1 = chat1->GetLastMessage();
	auto respon2 = chat2->GetLastMessage();
	printf("[0]===> %s\n", respon0);
	printf("[1]===> %s\n", respon1);
	printf("[2]===> %s\n", respon2);

	printf("Finished");
}

//========================================================================================================

void testLlamaMtmd()
{
	auto modelpath = "E:\\_models_\\llm_multimodel_gguf\\gemma-3-4b-it-Q8_0.gguf";
	auto mmprojpath = "E:\\_models_\\llm_multimodel_gguf\\gemma-3-4b-it-Q8_0-mmproj-model-f16.gguf";
	//auto modelpath = "E:\\_models_\\llm_gguf\\google_gemma-4-E4B-it-Q6_K_L.gguf";
	//auto mmprojpath = "E:\\_models_\\llm_gguf\\mmproj-google_gemma-4-E4B-it-f16.gguf";
	LLamaWrapper::Settings sets;
	sets.logEnabled = true;
	sets.samplingTemperature = 0.2f;
	sets.n_parallel = 2;
	sets.ctxSize = 4096;
	//params.mmproj_use_gpu = true;

	auto img0 = Image2d<uint8_t>(1, 1, { 1 }, ColorSpace::PixelFormat::GRAY);
	auto img1 = Image2d<uint8_t>(1, 1, { 1 }, ColorSpace::PixelFormat::GRAY);

	auto mm = LLamaChatRunnerMultiModal(modelpath, mmprojpath, sets);
	mm.InitModel();

	auto chat0 = mm.GetChatAs<LLamaChatMultiModal>(0);
	chat0->AddMessage("Whats in the image", img0);

	auto chat1 = mm.GetChatAs<LLamaChatMultiModal>(1);
	chat1->AddMessage("Whats on the image", img1);

	mm.Run();

	std::string response0 = chat0->GetLastMessage();
	printf("IMAGE0:\n %s\n", response0.c_str());
	std::string response1 = chat1->GetLastMessage();
	printf("IMAGE1:\n %s\n", response1.c_str());

	printf("Finished");
}

//========================================================================================================

void testGemma4CustomTemplate()
{
	LLamaWrapper::Settings setsAddr;
	setsAddr.ctxSize = 1024;
	//setsAddr.logicalBatchSize = setsAddr.ctxSize;
	//setsAddr.physicalBatchSize = setsAddr.ctxSize;
	setsAddr.gpuLayers = 999;
	setsAddr.logEnabled = true;
	setsAddr.n_parallel = 1;

	//https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF/tree/main

	auto chatRunner = std::make_shared<LLamaChatRunner>("E://_models_//llm_gguf//google_gemma-4-E4B-it-Q6_K_L.gguf", setsAddr);

	chatRunner->InitModel();

	chatRunner->SetCustomPromptBuilder([](const std::vector<llama_chat_message>& msgs) -> std::string {

		std::string prompt = "";
		for (const auto& m : msgs)
		{
			prompt += "<|turn>";
			prompt += (m.role == LLamaChat::ROLE_ASSISTANT) ? "model" : m.role;
			prompt += "\n";

			prompt += m.content;
			prompt += "<turn|>\n";
		}

		prompt += "<|turn>model\n";

		return prompt;
		});

	auto chat = chatRunner->GetChat(0);
	chat->AddMessage("What is car (simple, one sentence answer)");

	//for use with image in mtmd chat:
	//auto chat = chatRunner->GetChatAs<LLamaChatMultiModal>(0);
	//chat->AddMessage("<|image|>Whats in the image", img0);

	chatRunner->Run();

	auto response = chat->GetLastMessage();

	printf("\033[33m");
	printf("%s", response);
	printf("\n\033[0m");

	while (true)
	{
		std::cout << "Enter query:\n";
		std::string query;
		std::getline(std::cin, query);
		if (query == "exit")
		{
			break;
		}

		if (query == ":clear")
		{
			chat->Clear();
			continue;
		}

		chat->AddMessage(query);

		chatRunner->Run();

		auto response = chat->GetLastMessage();

		printf("\033[33m");
		printf("%s", response);
		printf("\n\033[0m");
	}

	printf("Finished");
}

//========================================================================================================



int main(int argc, char** argv)
{
	//testLlamaParallel();

	//testGemma4CustomTemplate();

	//testLlamaMtmd();

	{
		LLamaWrapper::Settings settings;
		settings.ctxSize = 16384;
		settings.physicalBatchSize = 2048;

		LLamaDecisionMultiModal decision(
			"E://_models_//llm_decision_gguf//vllm-d3-nano-Q8_0.gguf",
			"E://_models_//llm_decision_gguf//mmproj-vllm-d3-nano-Q8_0.gguf",
			settings
		);

		Question question{
			"color", QuestionType::CHOICE, "Which color is shown?",
			{{"black"}, {"white"}}
		};

		// A one-pixel RGB example. Supply your decoded image's dimensions and pixels.
		Image2d<uint8_t> image(1, 1, { 0,0,0 }, ColorSpace::PixelFormat::RGB);
		auto answers = decision.Run("Classify the supplied image.", { question }, { image });

		printf("mtmd");
	}

	LLamaWrapper::Settings settings;
	settings.ctxSize = 4096;
	LLamaDecision decision("E://_models_//llm_decision_gguf//Julia-1-Q8_0.gguf", settings);
	//LLamaDecision decision("E://_models_//llm_decision_gguf//vllm-d3-nano-Q8_0.gguf", settings);

	// Application-defined: emit the exact special tokens and layout your model expects.
	//decision.SetCustomPromptBuilder([](const LLamaDecision::PromptInput& input) {
	//	return BuildMyDecisionPrompt(input);
	//});

	std::vector<Question> questions = {
		{"sentiment", QuestionType::CHOICE, "Classify the sentiment.", {
			{"positive", "A favorable review"},
			{"negative", "An unfavorable review"}
		}},
		{"quality", QuestionType::SCORE, "Rate the quality.", {
			{"Poor"}, {"Average"}, {"Excellent"}
		}},
		{"satisfied", QuestionType::NOUL, "Is the customer satisfied?", {}}
	};
	auto answers = decision.Run("Excellent product, I am very happy.", questions);
	//auto answers = decision.Run("Vynikajici produkt, ale hned se rozbil. Musel jsem koupit novy.", questions);
	const DecisionAnswer& sentiment = answers.at("sentiment");
	float positiveProbability = sentiment.probabs[0];
	float confidence = sentiment.confidence;
	float satisfaction = answers.at("satisfied").confidence;


	return 0;
}