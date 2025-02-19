#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

enum class KernelSchedule
{
    WarpSpecialized,
    WarpSpecialized_Cooperative,
    WarpSpecialized_PingPong
};

enum class TileSchedule
{
    NO_SPLIT_K,
    SPLIT_K_SERIAL,
    STREAM_K_SK,
    STREAM_K_DP
};

enum class MainloopSchedule
{
    AUTO // Automatically selects between pingpong and cooperative schedules on Hopper. On older architectures, this
         // defaults to the "legacy" main loop schedule.
};

enum class EpilogueSchedule
{
    AUTO // Automatically chooses an epilogue schedule compatible with the selected main loop schedule for Hopper. For
         // architectures older than hopper, the epilogue is always performed by the same thread block as the main loop.
};

enum class ClusterShape
{
    ClusterShape_1x1x1,
    ClusterShape_2x1x1,
    ClusterShape_1x2x1,
    ClusterShape_2x2x1,
    ClusterShape_1x8x1,
    ClusterShape_8x1x1
};

enum class CutlassTileConfigSM90
{
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // CTA configs for M=64
    CtaShape64x16x128B,
    CtaShape64x32x128B,
    CtaShape64x64x128B,
    CtaShape64x128x128B,
    CtaShape64x256x128B,

    // CTA configs for M=128
    CtaShape128x16x128B,
    CtaShape128x32x128B,
    CtaShape128x64x128B,
    CtaShape128x128x128B,
    CtaShape128x256x128B,

    // CTA configs for M=128
    CtaShape256x128x128B,
};

struct CutlassGemmConfig
{
    // config options for sm90
    CutlassTileConfigSM90 tile_config = CutlassTileConfigSM90::ChooseWithHeuristic;
    MainloopSchedule mainloop_schedule = MainloopSchedule::AUTO;
    EpilogueSchedule epilogue_schedule = EpilogueSchedule::AUTO;
    ClusterShape cluster_shape = ClusterShape::ClusterShape_1x1x1;
    TileSchedule tile_schedule = TileSchedule::NO_SPLIT_K;

    CutlassGemmConfig(CutlassTileConfigSM90 tile_config_sm90,
                      MainloopSchedule      mainloop_schedule,
                      EpilogueSchedule      epilogue_schedule,
                      ClusterShape          cluster_shape,
                      TileSchedule          tile_schedule)
        : tile_config(tile_config_sm90)
        , mainloop_schedule(mainloop_schedule)
        , epilogue_schedule(epilogue_schedule)
        , cluster_shape(cluster_shape)
        , tile_schedule(tile_schedule)
    {

    }
};