use bevy::ecs::{
    schedule::{IntoScheduleConfigs, Schedule, ScheduleLabel, SystemSet},
    world::World,
};

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum RenderSystems {
    Begin,
    Render,
    End,
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
                RenderSystems::Begin,
                RenderSystems::Render,
                RenderSystems::End,
            )
                .chain(),
        );

        schedule
    }
}

pub fn run_render_startup_schedule(world: &mut World) {
    world.run_schedule(RenderStartup);
}

pub fn run_render_schedule(world: &mut World) {
    world.run_schedule(Render);
}
