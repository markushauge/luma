use std::collections::HashMap;

use anyhow::Result;
use bevy::{
    ecs::system::{StaticSystemParam, SystemParam, SystemParamItem},
    prelude::*,
};

pub trait RenderAsset: Send + Sync + Sized + 'static {
    type SourceAsset: Asset;
    type Param: SystemParam;

    fn prepare(
        source_asset: &Self::SourceAsset,
        param: &mut SystemParamItem<Self::Param>,
    ) -> Result<Self>;
}

#[derive(Resource)]
pub struct RenderAssets<A: RenderAsset> {
    assets: HashMap<AssetId<A::SourceAsset>, A>,
}

impl<A: RenderAsset> Default for RenderAssets<A> {
    fn default() -> Self {
        Self {
            assets: HashMap::new(),
        }
    }
}

impl<A: RenderAsset> RenderAssets<A> {
    pub fn get(&self, id: &AssetId<A::SourceAsset>) -> Option<&A> {
        self.assets.get(id)
    }

    pub fn insert(&mut self, id: AssetId<A::SourceAsset>, asset: A) -> Option<A> {
        self.assets.insert(id, asset)
    }

    pub fn remove(&mut self, id: &AssetId<A::SourceAsset>) -> Option<A> {
        self.assets.remove(id)
    }
}

pub fn sync_render_assets<A: RenderAsset>(
    mut render_assets: ResMut<RenderAssets<A>>,
    mut asset_events: MessageReader<AssetEvent<A::SourceAsset>>,
    assets: Res<Assets<A::SourceAsset>>,
    param: StaticSystemParam<A::Param>,
) -> Result<(), BevyError> {
    let mut param = param.into_inner();

    for event in asset_events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                if let Some(source) = assets.get(*id) {
                    let render_asset = A::prepare(source, &mut param)?;
                    render_assets.insert(*id, render_asset);
                }
            }
            AssetEvent::Removed { id } | AssetEvent::Unused { id } => {
                render_assets.remove(id);
            }
            _ => {}
        }
    }

    Ok(())
}
