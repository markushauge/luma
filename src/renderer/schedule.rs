use bevy::ecs::{
    schedule::{IntoScheduleConfigs, Schedule, ScheduleLabel, SystemSet},
    world::World,
};

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum RenderSystems {
    Prepare,
    PrepareAssets,
    PrepareRayTracing,
    Queue,
    QueueRayTracing,
    QueueUi,
    Submit,
}

#[derive(ScheduleLabel, Debug, Hash, PartialEq, Eq, Clone)]
pub struct RenderStartup;

impl RenderStartup {
    pub fn schedule() -> Schedule {
        Schedule::new(Self)
    }
}

#[derive(ScheduleLabel, Debug, Hash, PartialEq, Eq, Clone)]
pub struct Render;

impl Render {
    pub fn schedule() -> Schedule {
        let mut schedule = Schedule::new(Self);

        schedule.configure_sets(
            (
                RenderSystems::Prepare,
                RenderSystems::Queue,
                RenderSystems::Submit,
            )
                .chain(),
        );

        schedule.configure_sets(
            (
                RenderSystems::PrepareAssets,
                RenderSystems::PrepareRayTracing,
            )
                .chain()
                .in_set(RenderSystems::Prepare),
        );

        schedule.configure_sets(
            (RenderSystems::QueueRayTracing, RenderSystems::QueueUi)
                .chain()
                .in_set(RenderSystems::Queue),
        );

        schedule
    }
}

pub fn run_render_startup_schedule(world: &mut World) {
    world.run_schedule(RenderStartup);
}
