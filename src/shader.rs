use std::{ffi::CString, path::Path};

use anyhow::{Error, Result, anyhow};
use bevy::{
    asset::{AssetLoader, AsyncReadExt, LoadContext, io::Reader},
    prelude::*,
    tasks::ConditionalSendFuture,
};
use serde::{Deserialize, Serialize};

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

#[derive(Default, Serialize, Deserialize)]
pub struct ShaderSettings {
    pub entry_point: Option<String>,
}

pub struct ShaderLoader {
    compiler: shaderc::Compiler,
}

impl Default for ShaderLoader {
    fn default() -> Self {
        let compiler = shaderc::Compiler::new().unwrap();
        Self { compiler }
    }
}

impl AssetLoader for ShaderLoader {
    type Asset = Shader;
    type Settings = ShaderSettings;
    type Error = Error;

    fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        load_context: &mut LoadContext,
    ) -> impl ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let path = load_context
                .path()
                .to_str()
                .ok_or_else(|| anyhow!("Path is not a valid UTF-8 string"))?;

            let extension = load_context
                .path()
                .extension()
                .and_then(|extension| extension.to_str());

            let kind = match extension {
                Some("comp") => shaderc::ShaderKind::Compute,
                Some("rgen") => shaderc::ShaderKind::RayGeneration,
                Some("rmiss") => shaderc::ShaderKind::Miss,
                Some("rchit") => shaderc::ShaderKind::ClosestHit,
                Some("rahit") => shaderc::ShaderKind::AnyHit,
                Some("rint") => shaderc::ShaderKind::Intersection,
                Some("rcall") => shaderc::ShaderKind::Callable,
                _ => {
                    return Err(anyhow!(
                        "Could not infer shader kind from extension: {:?}",
                        extension
                    ));
                }
            };

            let mut source = String::new();
            reader.read_to_string(&mut source).await?;

            let mut options = shaderc::CompileOptions::new()?;
            options.set_include_callback(include_callback);

            let entry_point = settings.entry_point.as_deref().unwrap_or("main");

            let artifact = self.compiler.compile_into_spirv(
                &source,
                kind,
                path,
                entry_point,
                Some(&options),
            )?;

            if artifact.get_num_warnings() > 0 {
                tracing::warn!(
                    "Shader compilation warnings:\n{}",
                    artifact.get_warning_messages()
                );
            }

            let code = artifact.as_binary().to_vec();
            let entry_point = CString::new(entry_point)?;
            Ok(Shader { code, entry_point })
        })
    }
}

fn include_callback(
    requested_source: &str,
    include_type: shaderc::IncludeType,
    _requesting_source: &str,
    include_depth: usize,
) -> Result<shaderc::ResolvedInclude, String> {
    if include_depth > 10 {
        return Err("Include depth exceeded 10".to_owned());
    }

    let name = match include_type {
        shaderc::IncludeType::Relative => requested_source,
        shaderc::IncludeType::Standard => {
            return Err("Standard include type not supported".to_owned());
        }
    };

    // TODO: Use the asset server to resolve the path
    let parent = Path::new("assets/shaders");
    let resolved_path = parent.join(name);
    let content = std::fs::read_to_string(&resolved_path)
        .map_err(|e| format!("Failed to read included file: {e}"))?;

    Ok(shaderc::ResolvedInclude {
        resolved_name: resolved_path.to_string_lossy().to_string(),
        content,
    })
}
