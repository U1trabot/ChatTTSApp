use std::io::{self, Write};

use crate::synth::Synthesizer;
use pyo3::PyErr;
use rand;
use rodio::{buffer::SamplesBuffer, OutputStream, Sink};

use llm::{InferenceError, InferenceParameters, InferenceSession, KnownModel, ModelParameters};

pub enum TextMode {
    Instruction,
    Chat,
}

pub struct AiSession<M: KnownModel> {
    synth: Synthesizer,
    llm: M,
    session: InferenceSession,
    system: String,
    mode: TextMode,
    sink: Sink,
    _stream: OutputStream,
}

impl<M: KnownModel> AiSession<M> {
    pub fn new(
        model_path: &str,
        text_mode: TextMode,
        system_prompt: Option<String>,
    ) -> AiSession<M> {
        let synth = Synthesizer::new("tts_models/en/vctk/vits", false);
        let (_stream, stream_handle) = rodio::OutputStream::try_default().unwrap();
        let sink = rodio::Sink::try_new(&stream_handle).unwrap();
        sink.play();
        let model = llm::load::<M>(
            // path to GGML file
            std::path::Path::new(model_path),
            // llm::ModelParameters
            llm::TokenizerSource::Embedded,
            ModelParameters {
                use_gpu: true,
                ..Default::default()
            },
            // load progress callback
            llm::load_progress_callback_stdout,
        )
        .unwrap_or_else(|err| panic!("Failed to load model: {err}"));
        let session = model.start_session(Default::default());
        let sys_prompt = system_prompt.unwrap_or("Below is an instruction that describes a task. Write a response that appropriately completes the request.".to_string());
        AiSession {
            synth,
            llm: model,
            session,
            system: sys_prompt,
            mode: text_mode,
            sink,
            _stream,
        }
    }
    pub fn prompt(&mut self, prompt: &str) -> Result<String, InferenceError> {
        let mut text: String = String::new();

        let full_prompt = match &self.mode {
            TextMode::Instruction => format!(
                "<s>[INST] <<SYS>> {} <</SYS>> {} [/INST]",
                &self.system, prompt
            ),
            TextMode::Chat => format!(
                "### System: {} ### Human: {} ### Assistant: ",
                &self.system, prompt
            ),
        };
        let _res = &self.session.infer::<std::convert::Infallible>(
            // model to use for text generation
            &self.llm,
            // randomness provider
            &mut rand::thread_rng(),
            // the prompt to use for text generation, as well as other
            // inference parameters
            &llm::InferenceRequest {
                prompt: llm::Prompt::Text(full_prompt.as_str()),
                parameters: &InferenceParameters::default(),
                play_back_previous_tokens: false,
                maximum_token_count: Some(500usize),
            },
            // llm::OutputRequest
            &mut Default::default(),
            // output callback
            |t| {
                match t {
                    llm::InferenceResponse::SnapshotToken(_) => (),
                    llm::InferenceResponse::PromptToken(_) => (),
                    llm::InferenceResponse::InferredToken(str) => {
                        if !str.is_empty() {
                            text += &str
                        }
                    }
                    llm::InferenceResponse::EotToken => (),
                };

                Ok(llm::InferenceFeedback::Continue)
            },
        )?;
        Ok(text)
    }
    pub fn speak(&mut self, text: String) -> Result<(), PyErr> {
        let audio = self.synth.tts(&text)?;
        let rate = self.synth.sample_rate()?;
        println!("playing audio at rate {}", rate);
        let buff = SamplesBuffer::new(1, rate as u32, audio.clone());
        self.sink.append(buff);
        self.sink.sleep_until_end();
        Ok(io::stdout().flush().unwrap())
    }
}
