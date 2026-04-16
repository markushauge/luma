use std::{ffi::CString, fs::File, io::Read, path::Path, process::Command};

use anyhow::{Error, Ok, Result, anyhow};
use bevy::{
    asset::{AssetLoader, LoadContext, io::Reader},
    prelude::*,
    tasks::ConditionalSendFuture,
};

pub struct ShaderPlugin;

impl Plugin for ShaderPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Shader>()
            .init_asset_loader::<ShaderLoader>();
    }
}

#[derive(Asset, TypePath)]
pub struct Shader {
    pub code: Vec<u32>,
    pub entry_point: CString,
}

#[derive(TypePath, Default)]
pub struct ShaderLoader;

impl AssetLoader for ShaderLoader {
    type Asset = Shader;
    type Settings = ();
    type Error = Error;

    fn load(
        &self,
        _reader: &mut dyn Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext,
    ) -> impl ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let input_path = Path::new("assets").join(load_context.path().path());
            let output_path = input_path.with_extension("spv");
            let mut command = Command::new("slangc");

            command
                .arg(&input_path)
                .arg("-o")
                .arg(&output_path)
                .arg("-fvk-use-scalar-layout");

            let output = command.output()?;

            if output.stderr.len() > 0 {
                let stderr = std::str::from_utf8(&output.stderr)
                    .map_err(|e| anyhow!("Failed to parse shader compiler output: {e}"))?;

                return Err(anyhow!("{stderr}"));
            }

            let file = File::open(&output_path)
                .map_err(|e| anyhow!("Failed to open compiled shader: {e}"))?;

            let code =
                read_spv(file).map_err(|e| anyhow!("Failed to read compiled shader: {e}"))?;

            let entry_point = c"main".to_owned();
            Ok(Shader { code, entry_point })
        })
    }
}

fn read_spv(mut file: File) -> Result<Vec<u32>> {
    let len = file.metadata()?.len() as usize;

    if !len.is_multiple_of(4) {
        anyhow::bail!("SPIR-V bytecode length must be divisible by 4");
    }

    let words = len / 4;
    let mut code = vec![0; words];
    file.read_exact(bytemuck::cast_slice_mut(&mut code))?;
    Ok(code)
}
