use anyhow::{Error, Result};
use bevy::{
    asset::{AssetLoader, LoadContext, RenderAssetUsages, io::Reader},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

pub struct GltfPlugin;

impl Plugin for GltfPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Gltf>().init_asset_loader::<GltfLoader>();
    }
}

#[expect(unused)]
#[derive(Asset, TypePath)]
pub struct Gltf {
    pub meshes: Vec<Handle<Mesh>>,
}

#[derive(Default, TypePath)]
pub struct GltfLoader;

impl AssetLoader for GltfLoader {
    type Asset = Gltf;
    type Settings = ();
    type Error = Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Gltf> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;

        let gltf = gltf::Gltf::from_slice(&bytes)?;
        let blob = gltf.blob.as_deref();
        let mut meshes = Vec::new();

        for (mesh_index, gltf_mesh) in gltf.document.meshes().enumerate() {
            for (primitive_index, primitive) in gltf_mesh.primitives().enumerate() {
                let reader = primitive.reader(|buffer| match buffer.source() {
                    gltf::buffer::Source::Bin => blob,
                    gltf::buffer::Source::Uri(_) => None,
                });

                let mut mesh = Mesh::new(
                    PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                );

                if let Some(positions) = reader.read_positions() {
                    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.collect::<Vec<_>>());
                }

                if let Some(normals) = reader.read_normals() {
                    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals.collect::<Vec<_>>());
                }

                if let Some(uvs) = reader.read_tex_coords(0) {
                    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs.into_f32().collect::<Vec<_>>());
                }

                if let Some(indices) = reader.read_indices() {
                    mesh.insert_indices(Indices::U32(indices.into_u32().collect()));
                }

                let label = format!("Mesh{mesh_index}/Primitive{primitive_index}");
                meshes.push(load_context.add_labeled_asset(label, mesh));
            }
        }

        Ok(Gltf { meshes })
    }

    fn extensions(&self) -> &[&str] {
        &["gltf", "glb"]
    }
}
