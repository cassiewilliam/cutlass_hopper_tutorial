#pragma once

namespace cutlass_w4a8
{

template<typename OutType,
         int      kStages,
         typename TileShape,
         typename ClusterShape>
struct MixedInputGemmKernelTraits
{
    using MmaType   = cutlass::float_e4m3_t;
    using QuantType = cutlass::int4b_t;
    using ScaleType = OutType;

    static constexpr int per_group_size = 128;

    /// GEMM with TN support
    // A matrix configuration
    // NOTE(Alan): A is Weight
    using ElementA = QuantType;                      // Element type for B matrix operand
    using LayoutA  = cutlass::layout::ColumnMajor;   // Layout type for B matrix operand

    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    // B matrix configuration
    // NOTE(Alan): B is Activateion
    using ElementB = MmaType;
    using LayoutB  = cutlass::layout::RowMajor;   // Layout type for A matrix operand

    // Alignment of A matrix in units of elements (up to 16 bytes)
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    // Define the CuTe layout for reoredered quantized tensor A
    // LayoutAtomQuant places values that will be read by the same thread in contiguous locations in
    // global memory. It specifies the reordering within a single warp's fragment
    using LayoutBtomQuant   = decltype(cutlass::compute_memory_reordering_atom<ElementB>());

    using LayoutA_Reordered = decltype(cute::tile_to_shape(
        LayoutBtomQuant{},
        cute::Layout<cute::Shape<int, int, int>, cutlass::detail::TagToStrideA_t<LayoutA>>{}));

    // This example manually swaps and transposes, so keep transpose of input layouts
    using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
};



}