

```cpp
template <class ArchTag, class OpClass,
          class ElementA, class GmemLayoutA, int AlignmentA,
          class ElementB, class GmemLayoutB, int AlignmentB,
          class ElementAccumulator,
          class TileShape_MNK,
          class ClusterShape_MNK,
          class StageCountType,
          class KernelScheduleType,
          class Enable = void>
struct CollectiveBuilder {
  static_assert(sizeof(ElementA) == 0, "Could not build a collective for given parameters.");
};

// GMMA_TMA_WS_SS
template <class ElementA, class GmemLayoutATag, int AlignmentA,
          class ElementB, class GmemLayoutBTag, int AlignmentB,
          class ElementAccumulator,
          class TileShape_MNK,
          class ClusterShape_MNK,
          class StageCountType,
          class KernelScheduleType>
struct CollectiveBuilder<arch::Sm90, arch::OpClassTensorOp,
                         ElementA, GmemLayoutATag, AlignmentA,
                         ElementB, GmemLayoutBTag, AlignmentB,
                         ElementAccumulator,
                         TileShape_MNK,
                         ClusterShape_MNK,
                         StageCountType,
                         KernelScheduleType,
                         cute::enable_if_t<
                            (cute::is_any_of_v<KernelScheduleType,
                                               KernelTmaWarpSpecialized,
                                               KernelTmaWarpSpecializedCooperative,
                                               KernelTmaWarpSpecializedPingpong,
                                               KernelPtrArrayTmaWarpSpecializedCooperative,
                                               KernelPtrArrayTmaWarpSpecializedPingpong>) &&
                             not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
                        >
{

}

// GMMA_TMA_WS_RS 
template <class ElementA_, class GmemLayoutATag_, int AlignmentA,
          class ElementB_, class GmemLayoutBTag_, int AlignmentB,
          class ElementAccumulator,
          class TileShape_MNK,
          class ClusterShape_MNK,
          class StageCountType,
          class KernelScheduleType>
struct CollectiveBuilder<arch::Sm90, arch::OpClassTensorOp,
                         ElementA_, GmemLayoutATag_, AlignmentA,
                         ElementB_, GmemLayoutBTag_, AlignmentB,
                         ElementAccumulator,
                         TileShape_MNK,
                         ClusterShape_MNK,
                         StageCountType,
                         KernelScheduleType,
                         cute::enable_if_t<
                            (cute::is_same_v<KernelScheduleType,  KernelTmaWarpSpecialized> ||
                             cute::is_same_v<KernelScheduleType,  KernelTmaWarpSpecializedPingpong> ||
                             cute::is_same_v<KernelScheduleType,  KernelTmaWarpSpecializedCooperative> ||
                             cute::is_same_v<KernelScheduleType,  KernelPtrArrayTmaWarpSpecializedCooperative> ||
                             cute::is_same_v<KernelScheduleType,  KernelPtrArrayTmaWarpSpecializedPingpong>) && 
                            (detail::is_use_rmem_A<ElementA_, GmemLayoutATag_, ElementB_, GmemLayoutBTag_>() ||
                             // ConvertAndScale and ConvertAndScaleWithZero 
                             cute::is_tuple<ElementA_>::value || cute::is_tuple<ElementB_>::value || 
                             // DirectConvert
                             sizeof_bits<ElementA_>::value != sizeof_bits<ElementB_>::value)>
                        >
{
}

// GMMA_TMA_WS_FP8_FAST_ACCUM_SS
template <class ElementA, class GmemLayoutATag, int AlignmentA,
          class ElementB, class GmemLayoutBTag, int AlignmentB,
          class ElementAccumulator,
          class TileShape_MNK,
          class ClusterShape_MNK,
          class StageCountType,
          class KernelScheduleType>
struct CollectiveBuilder<arch::Sm90, arch::OpClassTensorOp,
                         ElementA, GmemLayoutATag, AlignmentA,
                         ElementB, GmemLayoutBTag, AlignmentB,
                         ElementAccumulator,
                         TileShape_MNK,
                         ClusterShape_MNK,
                         StageCountType,
                         KernelScheduleType,
                         cute::enable_if_t<
                            cute::is_any_of_v<KernelScheduleType,
                                            KernelTmaWarpSpecializedFP8FastAccum,
                                            KernelTmaWarpSpecializedPingpongFP8FastAccum,
                                            KernelTmaWarpSpecializedCooperativeFP8FastAccum,
                                            KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,
                                            KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum>>
                        >
{
}

// GMMA_TMA_SS
template <class ElementA, class GmemLayoutATag, int AlignmentA,
          class ElementB, class GmemLayoutBTag, int AlignmentB,
          class ElementAccumulator,
          class TileShape_MNK,
          class ClusterShape_MNK,
          class StageCountType,
          class KernelScheduleType>
struct CollectiveBuilder<arch::Sm90, arch::OpClassTensorOp,
                         ElementA, GmemLayoutATag, AlignmentA,
                         ElementB, GmemLayoutBTag, AlignmentB,
                         ElementAccumulator,
                         TileShape_MNK,
                         ClusterShape_MNK,
                         StageCountType,
                         KernelScheduleType,
                         cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelTma> &&
                            not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
                        >
{
}

// GMMA_CpAsync_WS_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int   AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int   AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedPingpong>) &&
      not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()
    >
> 
{
}

// GMMA_CpAsync_WS_RS
template <
  class ElementA,
  class GmemLayoutATag,
  int   AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int   AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedPingpong>) &&
      detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()
    >
> 
{
}

// GMMA auto kernel schedule
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelScheduleAuto>>
> 
{
}

// GMMA_TMA_WS_SS (BlockScaled Builders)
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  int ScaleGranularityM_
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum<ScaleGranularityM_>,
    cute::enable_if_t<
      not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
>
{
}
```