//! Rust bindings for the coqui-TTS python library for Text-To-Speech
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

/// TTS Synthesizer. equivilant to `TTS.utils.synthesizer.Synthesizer`
#[derive(Debug)]
pub struct Synthesizer {
    locals: Py<PyDict>,
}

impl Synthesizer {
    /// Create a new Synthesizer, performing startup initialization (this method is NOT cheap to call, expect a few SECONDS of runtime)
    ///
    /// this will also download apropreate models if they are missing
    ///
    /// # Arguments
    ///
    /// model: the name of the TTS model to use. see https://github.com/coqui-ai/TTS for models.
    ///
    /// # Note
    ///
    /// this may spew out some text to stdout about initialization,
    /// this is from the python library and there is nothing that can be done about it
    ///
    pub fn new(model: &str, use_cuda: bool) -> Self {
        Python::with_gil(|py| {
            let locals: Py<PyDict> = PyDict::new(py).into();
            let locals_ref = locals.as_ref(py);
            locals_ref.set_item("model_name", model).unwrap();
            locals_ref.set_item("use_cuda", use_cuda).unwrap();
            print!("Model name is: {}", model);
            match py.run(
                r#"
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
# create instance of the coqui tts model manager
manager = ModelManager()
# download the model
(
    model_path,
    config_path,
    model_item,
) = manager.download_model(model_name)
# download the vocoder
if model_item["default_vocoder"] is not None:
    vocoder_path, vocoder_config_path, _ = manager.download_model(
        model_item["default_vocoder"]
    )
else:
    vocoder_path = vocoder_config_path = None
# create the coqui tts instance
coqui_tts = Synthesizer(
    model_path,
    config_path,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    use_cuda=use_cuda
)
            "#,
                None,
                Some(locals.as_ref(py)),
            ) {
                Ok(_) => (),
                Err(p) => {
                    if let Some(trace) = p.traceback(py) {
                        panic!("{}: {}", p, trace.format().unwrap())
                    }
                }
            }

            Self { locals }
        })
    }

    /// Synthesize some audio.
    ///
    /// # Returned format
    /// channels: 1?
    /// rate: see [`Synthesizer::sample_rate`]
    ///
    pub fn tts(&mut self, text: &str) -> Result<Vec<f32>, PyErr> {
        Python::with_gil(|py| {
            let tts = self.locals.as_ref(py).get_item("coqui_tts")?.unwrap();
            let audio = tts
                .call_method1("tts", (text, String::from("p225")))
                .unwrap()
                .downcast::<PyList>()
                .unwrap();
            Ok(audio.extract::<Vec<f32>>().unwrap())
        })
    }

    pub fn sample_rate(&mut self) -> Result<u64, PyErr> {
        Python::with_gil(|py| {
            let tts = self.locals.as_ref(py).get_item("coqui_tts")?.unwrap();
            Ok(tts
                .getattr("output_sample_rate")
                .unwrap()
                .extract::<u64>()
                .unwrap())
        })
    }
}
