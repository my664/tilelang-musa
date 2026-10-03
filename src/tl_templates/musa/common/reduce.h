#pragma once

#include "intrin.h"

#include <cstdint>
#include <type_traits>

namespace tl {

template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op);

template <typename T> struct AccType {
  using type = T;
};

#ifdef TL_MUSA_ENABLE_FP16
template <> struct AccType<half> {
  using type = float;
};
#endif

#ifdef TL_MUSA_ENABLE_BF16
template <> struct AccType<mt_bfloat16> {
  using type = float;
};
#endif

struct SumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }
};

struct MaxOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x > y ? x : y;
  }

#ifdef TL_MUSA_ENABLE_FP16
  TL_DEVICE half operator()(half const &x, half const &y) {
    return __hmax(x, y);
  }
#endif

#ifdef TL_MUSA_ENABLE_BF16
  TL_DEVICE mt_bfloat16 operator()(mt_bfloat16 const &x,
                                  mt_bfloat16 const &y) {
    return __hmax(x, y);
  }
#endif
};

struct MinOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x < y ? x : y;
  }

#ifdef TL_MUSA_ENABLE_FP16
  TL_DEVICE half operator()(half const &x, half const &y) {
    return __hmin(x, y);
  }
#endif

#ifdef TL_MUSA_ENABLE_BF16
  TL_DEVICE mt_bfloat16 operator()(mt_bfloat16 const &x,
                                  mt_bfloat16 const &y) {
    return __hmin(x, y);
  }
#endif
};

struct BitAndOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x & y;
  }
};

struct BitOrOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x | y;
  }
};

struct BitXorOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x ^ y;
  }
};

struct SyncThreadsBarrier {
  template <int phase = 0> static TL_DEVICE void sync() { __syncthreads(); }
};

template <typename T>
TL_DEVICE T shfl_xor_sync(unsigned mask, T val, int lane_mask) {
  return __shfl_xor_sync(mask, val, lane_mask);
}

TL_DEVICE float2 shfl_xor_sync(unsigned mask, float2 val, int lane_mask) {
  float2 out;
  out.x = __shfl_xor_sync(mask, val.x, lane_mask);
  out.y = __shfl_xor_sync(mask, val.y, lane_mask);
  return out;
}

#ifdef TL_MUSA_ENABLE_FP16
TL_DEVICE half shfl_xor_sync(unsigned mask, half val, int lane_mask) {
  float raw = static_cast<float>(val);
  return half(__shfl_xor_sync(mask, raw, lane_mask));
}
#endif

#ifdef TL_MUSA_ENABLE_BF16
TL_DEVICE mt_bfloat16 shfl_xor_sync(unsigned mask, mt_bfloat16 val,
                                   int lane_mask) {
  float raw = static_cast<float>(val);
  return mt_bfloat16(__shfl_xor_sync(mask, raw, lane_mask));
}
#endif

template <class Reducer, int threads, int scale, int thread_offset = 0,
          class Barrier = SyncThreadsBarrier, int batch_size = 1,
          int workspace_stride = 0>
struct AllReduce {
  static_assert(threads > 0, "tl::AllReduce threads must be positive");
  static_assert(scale > 0, "tl::AllReduce scale must be positive");
  static_assert(threads % scale == 0,
                "tl::AllReduce threads must be divisible by scale");
  static_assert(((threads / scale) & (threads / scale - 1)) == 0,
                "tl::AllReduce reduce width must be a power of two");

  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return x;
    } else {
      return butterfly_reduce_scalar(x, red_buf);
    }
  }

  template <typename T>
  static TL_DEVICE void run_batch(T *x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return;
    } else {
      butterfly_reduce_batch(x, red_buf);
    }
  }

private:
  using Next = AllReduce<Reducer, threads / 2, scale, thread_offset, Barrier,
                         batch_size, workspace_stride>;

  template <typename T>
  static TL_DEVICE T butterfly_reduce_scalar(T x, T *red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      Barrier::template sync<1>();
      red_buf[threadIdx.x - thread_offset] = x;
      Barrier::template sync<2>();
      x = Reducer()(x, red_buf[(threadIdx.x - thread_offset) ^ offset]);
    } else {
      x = Reducer()(x, tl::shfl_xor_sync(uint32_t(-1), x, offset));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return Next::run(x, red_buf);
    }
  }

  template <typename T>
  static TL_DEVICE void butterfly_reduce_batch(T *x, T *red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      Barrier::template sync<1>();
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        red_buf[(threadIdx.x - thread_offset) + i * workspace_stride] = x[i];
      }
      Barrier::template sync<2>();
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        x[i] =
            Reducer()(x[i], red_buf[((threadIdx.x - thread_offset) ^ offset) +
                                    i * workspace_stride]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        x[i] = Reducer()(x[i], tl::shfl_xor_sync(uint32_t(-1), x[i], offset));
      }
    }
    if constexpr (offset == scale) {
      return;
    } else {
      Next::run_batch(x, red_buf);
    }
  }
};

// Full-block, scale-1 reduction that performs one warp-local reduction per
// warp and one cross-warp reduction.  The ordinary AllReduce above is kept for
// scaled/partial reductions because its XOR topology is part of their layout
// contract.  This variant is selected by the MUSA lowering only when every
// thread in the block participates and the reduction scale is one.
template <class Reducer, int threads, int thread_offset = 0,
          class Barrier = SyncThreadsBarrier, int batch_size = 1,
          int workspace_stride = 0>
struct WarpFirstAllReduce {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ <= 220)
  static constexpr int warp_size = 128;
#else
  static constexpr int warp_size = 32;
#endif
  static constexpr int num_warps = threads / warp_size;

  static_assert(threads > warp_size,
                "WarpFirstAllReduce requires more than one warp");
  static_assert(threads % warp_size == 0,
                "WarpFirstAllReduce threads must be warp aligned");
  // The cross-warp shuffle mask is a 32-bit lane mask on the supported MUSA
  // targets, so the second-stage reduction must fit in 32 lanes.
  static_assert(num_warps <= 32,
                "WarpFirstAllReduce cross-warp reduction exceeds mask width");
  static_assert((num_warps & (num_warps - 1)) == 0,
                "WarpFirstAllReduce warp count must be a power of two");
  static_assert(batch_size == 1 || workspace_stride > 0,
                "WarpFirstAllReduce batched workspace must have a stride");

  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    const int local_thread = threadIdx.x - thread_offset;
    const int lane = local_thread % warp_size;
    const int warp = local_thread / warp_size;
    const T warp_value = tl::warp_reduce<T>(x, Reducer());

    if (lane == 0) {
      red_buf[warp] = warp_value;
    }
    Barrier::template sync<1>();

    if (warp == 0 && lane < num_warps) {
      constexpr unsigned mask =
          num_warps >= 32 ? 0xffffffffu : ((1u << num_warps) - 1u);
      T total = red_buf[lane];
      for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
        total = Reducer()(total, tl::shfl_xor_sync(mask, total, offset));
      }
      if (lane == 0) {
        red_buf[0] = total;
      }
    }
    Barrier::template sync<2>();
    return red_buf[0];
  }

  template <typename T>
  static TL_DEVICE void run_batch(T *x, T *red_buf = nullptr) {
    const int local_thread = threadIdx.x - thread_offset;
    const int lane = local_thread % warp_size;
    const int warp = local_thread / warp_size;

#pragma unroll
    for (int i = 0; i < batch_size; ++i) {
      x[i] = tl::warp_reduce<T>(x[i], Reducer());
      if (lane == 0) {
        red_buf[warp + i * workspace_stride] = x[i];
      }
    }
    Barrier::template sync<1>();

    if (warp == 0 && lane < num_warps) {
      constexpr unsigned mask =
          num_warps >= 32 ? 0xffffffffu : ((1u << num_warps) - 1u);
#pragma unroll
      for (int i = 0; i < batch_size; ++i) {
        T total = red_buf[lane + i * workspace_stride];
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
          total =
              Reducer()(total, tl::shfl_xor_sync(mask, total, offset));
        }
        if (lane == 0) {
          red_buf[i * workspace_stride] = total;
        }
      }
    }
    Barrier::template sync<2>();

#pragma unroll
    for (int i = 0; i < batch_size; ++i) {
      x[i] = red_buf[i * workspace_stride];
    }
  }
};

template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op) {
  constexpr uint32_t mask = 0xffffffff;
  int warp_size = detail::default_warp_size();
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    value = op(value, tl::shfl_xor_sync(mask, value, offset));
  }
  return value;
}

template <typename T> TL_DEVICE T warp_reduce_sum(T value) {
  return warp_reduce<T>(value, SumOp());
}

template <typename T> TL_DEVICE T warp_reduce_max(T value) {
  return warp_reduce<T>(value, MaxOp());
}

template <typename T> TL_DEVICE T warp_reduce_min(T value) {
  return warp_reduce<T>(value, MinOp());
}

template <typename T> TL_DEVICE T warp_reduce_bitand(T value) {
  return warp_reduce<T>(value, BitAndOp());
}

template <typename T> TL_DEVICE T warp_reduce_bitor(T value) {
  return warp_reduce<T>(value, BitOrOp());
}

} // namespace tl
