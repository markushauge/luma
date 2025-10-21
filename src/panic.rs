use native_dialog::{DialogBuilder, MessageLevel};

pub fn init_hook() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("{}", info);

        let _ = DialogBuilder::message()
            .set_level(MessageLevel::Error)
            .set_title("Panic")
            .set_text(info)
            .alert()
            .show();
    }));
}
