use std::{borrow::Cow, path::Path};

use anyhow::Result;

pub struct ShaderCompiler {
    compiler: shaderc::Compiler,
    options: shaderc::CompileOptions<'static>,
}

impl ShaderCompiler {
    pub fn new() -> Result<Self> {
        let compiler = shaderc::Compiler::new()?;
        let mut options = shaderc::CompileOptions::new()?;
        options.set_include_callback(include_callback);
        Ok(ShaderCompiler { compiler, options })
    }

    pub fn compile_file(&self, path: &str) -> Result<Vec<u32>> {
        let extension = Path::new(path)
            .extension()
            .and_then(|extension| extension.to_str());

        let kind = match extension {
            Some("comp") => shaderc::ShaderKind::Compute,
            _ => {
                return Err(anyhow::anyhow!(
                    "Could not infer shader kind from extension: {:?}",
                    extension
                ))
            }
        };

        let source = std::fs::read_to_string(path)?;
        self.compile_source(&source, kind, path)
    }

    pub fn compile_source(
        &self,
        source: &str,
        kind: shaderc::ShaderKind,
        filename: &str,
    ) -> Result<Vec<u32>> {
        let binary_result = self.compiler.compile_into_spirv(
            source,
            kind,
            filename,
            "main",
            Some(&self.options),
        )?;

        if binary_result.get_num_warnings() > 0 {
            tracing::warn!(
                "Shader compilation warnings:\n{}",
                binary_result.get_warning_messages()
            );
        }

        Ok(binary_result.as_binary().to_vec())
    }
}

fn include_callback(
    requested_source: &str,
    include_type: shaderc::IncludeType,
    requesting_source: &str,
    include_depth: usize,
) -> Result<shaderc::ResolvedInclude, String> {
    if include_depth > 10 {
        return Err("Include depth exceeded 10".to_owned());
    }

    let name = match include_type {
        shaderc::IncludeType::Relative => Cow::Borrowed(requested_source),
        shaderc::IncludeType::Standard => Cow::Owned(format!("{requested_source}.glsl")),
    };

    let parent = Path::new(requesting_source)
        .parent()
        .ok_or_else(|| "Could not determine parent directory of requesting source".to_owned())?;

    let resolved_path = parent.join(name.as_ref());
    let content = std::fs::read_to_string(&resolved_path)
        .map_err(|e| format!("Failed to read included file: {e}"))?;

    Ok(shaderc::ResolvedInclude {
        resolved_name: resolved_path.to_string_lossy().to_string(),
        content,
    })
}
