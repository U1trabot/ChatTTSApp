mod session;
mod synth;
use std::io::{self, Write};

use llm::models::Llama;
use session::{AiSession, TextMode};

fn main() -> io::Result<()> {
    let mut ai: AiSession<Llama> = AiSession::new(
        "models/cria-llama2-7b-v1.3.ggmlv3.q4_0.bin",
        TextMode::Instruction,
        Some("Below is an instruction that describes a task. Write a response that appropriately completes the request. In this response, you must talk like a depressed pirate".to_string()),
    );
    // Start main loop
    let mut buffer = String::new();
    let stdin = io::stdin();
    loop {
        println!();
        print!(">>> ");
        io::stdout().flush().expect("Could not flush buffer: ");
        stdin.read_line(&mut buffer)?;
        if buffer == "exit\n" {
            break;
        }
        if let Ok(resp) = ai.prompt(&buffer) {
            println!("{resp}");
            ai.speak(resp).expect("cannot");
            buffer.clear();
        } else {
            break;
        }
    }

    Ok(())
}
