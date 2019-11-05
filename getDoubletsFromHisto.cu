// -*- C++ -*-

// nvcc -std=c++14 --expt-relaxed-constexpr -gencode arch=compute_70,code=sm_70 --expt-extended-lambda -o getDoubletsFromHisto getDoubletsFromHisto.cu

#define NDEBUG

#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <iostream>
#include <limits>
#include <sstream>

#include <cuda_runtime.h>

namespace {

  [[noreturn]] inline void abortOnCudaError(
      const char* file, int line, const char* cmd, const char* error, const char* message) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "cudaCheck(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    std::cerr << out.str() << std::flush;
    abort();
  }

}  // namespace

inline bool cudaCheck_(const char* file, int line, const char* cmd, cudaError_t result) {
  if (__builtin_expect(result == cudaSuccess, true))
    return true;

  const char* error = cudaGetErrorName(result);
  const char* message = cudaGetErrorString(result);
  abortOnCudaError(file, line, cmd, error, message);
  return false;
}

#define cudaCheck(ARG) (cudaCheck_(__FILE__, __LINE__, #ARG, (ARG)))

namespace pixelGPUConstants {
  constexpr uint32_t maxNumberOfHits =
      48 * 1024;  // data at pileup 50 has 18300 +/- 3500 hits; 40000 is around 6 sigma away
}  // namespace pixelGPUConstants

namespace gpuClustering {
  // optimized for real data PU 50
  constexpr uint32_t maxHitsInIter() { return 160; }
  constexpr uint32_t maxHitsInModule() { return 1024; }

  constexpr uint32_t MaxNumModules = 2000;
  constexpr int32_t MaxNumClustersPerModules = maxHitsInModule();
  constexpr uint32_t MaxHitsInModule = maxHitsInModule();  // as above
  constexpr uint32_t MaxNumClusters = pixelGPUConstants::maxNumberOfHits;
  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

}  // namespace gpuClustering

namespace phase1PixelTopology {

  constexpr uint16_t numRowsInRoc = 80;
  constexpr uint16_t numColsInRoc = 52;
  constexpr uint16_t lastRowInRoc = numRowsInRoc - 1;
  constexpr uint16_t lastColInRoc = numColsInRoc - 1;

  constexpr uint16_t numRowsInModule = 2 * numRowsInRoc;
  constexpr uint16_t numColsInModule = 8 * numColsInRoc;
  constexpr uint16_t lastRowInModule = numRowsInModule - 1;
  constexpr uint16_t lastColInModule = numColsInModule - 1;

  constexpr int16_t xOffset = -81;
  constexpr int16_t yOffset = -54 * 4;

  constexpr uint32_t numPixsInModule = uint32_t(numRowsInModule) * uint32_t(numColsInModule);

  constexpr uint32_t numberOfModules = 1856;
  constexpr uint32_t numberOfLayers = 10;
  constexpr uint32_t layerStart[numberOfLayers + 1] = {0,
                                                       96,
                                                       320,
                                                       672,  // barrel
                                                       1184,
                                                       1296,
                                                       1408,  // positive endcap
                                                       1520,
                                                       1632,
                                                       1744,  // negative endcap
                                                       numberOfModules};
  constexpr char const* layerName[numberOfLayers] = {
      "BL1",
      "BL2",
      "BL3",
      "BL4",  // barrel
      "E+1",
      "E+2",
      "E+3",  // positive endcap
      "E-1",
      "E-2",
      "E-3"  // negative endcap
  };

  constexpr uint32_t numberOfModulesInBarrel = 1184;
  constexpr uint32_t numberOfLaddersInBarrel = numberOfModulesInBarrel / 8;

  template <class Function, std::size_t... Indices>
  constexpr auto map_to_array_helper(Function f, std::index_sequence<Indices...>)
      -> std::array<typename std::result_of<Function(std::size_t)>::type, sizeof...(Indices)> {
    return {{f(Indices)...}};
  }

  template <int N, class Function>
  constexpr auto map_to_array(Function f) -> std::array<typename std::result_of<Function(std::size_t)>::type, N> {
    return map_to_array_helper(f, std::make_index_sequence<N>{});
  }

  constexpr uint32_t findMaxModuleStride() {
    bool go = true;
    int n = 2;
    while (go) {
      for (uint8_t i = 1; i < 11; ++i) {
        if (layerStart[i] % n != 0) {
          go = false;
          break;
        }
      }
      if (!go)
        break;
      n *= 2;
    }
    return n / 2;
  }

  constexpr uint32_t maxModuleStride = findMaxModuleStride();

  constexpr uint8_t findLayer(uint32_t detId) {
    for (uint8_t i = 0; i < 11; ++i)
      if (detId < layerStart[i + 1])
        return i;
    return 11;
  }

  constexpr uint8_t findLayerFromCompact(uint32_t detId) {
    detId *= maxModuleStride;
    for (uint8_t i = 0; i < 11; ++i)
      if (detId < layerStart[i + 1])
        return i;
    return 11;
  }

  constexpr uint32_t layerIndexSize = numberOfModules / maxModuleStride;
  constexpr std::array<uint8_t, layerIndexSize> layer = map_to_array<layerIndexSize>(findLayerFromCompact);

  constexpr bool validateLayerIndex() {
    bool res = true;
    for (auto i = 0U; i < numberOfModules; ++i) {
      auto j = i / maxModuleStride;
      res &= (layer[j] < 10);
      res &= (i >= layerStart[layer[j]]);
      res &= (i < layerStart[layer[j] + 1]);
    }
    return res;
  }

  static_assert(validateLayerIndex(), "layer from detIndex algo is buggy");

  // this is for the ROC n<512 (upgrade 1024)
  constexpr inline uint16_t divu52(uint16_t n) {
    n = n >> 2;
    uint16_t q = (n >> 1) + (n >> 4);
    q = q + (q >> 4) + (q >> 5);
    q = q >> 3;
    uint16_t r = n - q * 13;
    return q + ((r + 3) >> 4);
  }

  constexpr inline bool isEdgeX(uint16_t px) { return (px == 0) | (px == lastRowInModule); }

  constexpr inline bool isEdgeY(uint16_t py) { return (py == 0) | (py == lastColInModule); }

  constexpr inline uint16_t toRocX(uint16_t px) { return (px < numRowsInRoc) ? px : px - numRowsInRoc; }

  constexpr inline uint16_t toRocY(uint16_t py) {
    auto roc = divu52(py);
    return py - 52 * roc;
  }

  constexpr inline bool isBigPixX(uint16_t px) { return (px == 79) | (px == 80); }

  constexpr inline bool isBigPixY(uint16_t py) {
    auto ly = toRocY(py);
    return (ly == 0) | (ly == lastColInRoc);
  }

  constexpr inline uint16_t localX(uint16_t px) {
    auto shift = 0;
    if (px > lastRowInRoc)
      shift += 1;
    if (px > numRowsInRoc)
      shift += 1;
    return px + shift;
  }

  constexpr inline uint16_t localY(uint16_t py) {
    auto roc = divu52(py);
    auto shift = 2 * roc;
    auto yInRoc = py - 52 * roc;
    if (yInRoc > 0)
      shift += 1;
    return py + shift;
  }

  //FIXME move it elsewhere?
  struct AverageGeometry {
    static constexpr auto numberOfLaddersInBarrel = phase1PixelTopology::numberOfLaddersInBarrel;
    float ladderZ[numberOfLaddersInBarrel];
    float ladderX[numberOfLaddersInBarrel];
    float ladderY[numberOfLaddersInBarrel];
    float ladderR[numberOfLaddersInBarrel];
    float ladderMinZ[numberOfLaddersInBarrel];
    float ladderMaxZ[numberOfLaddersInBarrel];
    float endCapZ[2];  // just for pos and neg Layer1
  };

}  // namespace phase1PixelTopology

namespace GPU {
  template <class T>
  struct SimpleVector {
    constexpr SimpleVector() = default;

    // ownership of m_data stays within the caller
    constexpr void construct(int capacity, T *data) {
      m_size = 0;
      m_capacity = capacity;
      m_data = data;
    }

    inline constexpr int push_back_unsafe(const T &element) {
      auto previousSize = m_size;
      m_size++;
      if (previousSize < m_capacity) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        --m_size;
        return -1;
      }
    }

    template <class... Ts>
    constexpr int emplace_back_unsafe(Ts &&... args) {
      auto previousSize = m_size;
      m_size++;
      if (previousSize < m_capacity) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        --m_size;
        return -1;
      }
    }

    __device__ inline T &back() { return m_data[m_size - 1]; }

    __device__ inline const T &back() const {
      if (m_size > 0) {
        return m_data[m_size - 1];
      } else
        return T();  //undefined behaviour
    }

    // thread-safe version of the vector, when used in a CUDA kernel
    __device__ int push_back(const T &element) {
      auto previousSize = atomicAdd(&m_size, 1);
      if (previousSize < m_capacity) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        atomicSub(&m_size, 1);
        return -1;
      }
    }

    template <class... Ts>
    __device__ int emplace_back(Ts &&... args) {
      auto previousSize = atomicAdd(&m_size, 1);
      if (previousSize < m_capacity) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        atomicSub(&m_size, 1);
        return -1;
      }
    }

    // thread safe version of resize
    __device__ int extend(int size = 1) {
      auto previousSize = atomicAdd(&m_size, size);
      if (previousSize < m_capacity) {
        return previousSize;
      } else {
        atomicSub(&m_size, size);
        return -1;
      }
    }

    __device__ int shrink(int size = 1) {
      auto previousSize = atomicSub(&m_size, size);
      if (previousSize >= size) {
        return previousSize - size;
      } else {
        atomicAdd(&m_size, size);
        return -1;
      }
    }

    inline constexpr bool empty() const { return m_size <= 0; }
    inline constexpr bool full() const { return m_size >= m_capacity; }
    inline constexpr T &operator[](int i) { return m_data[i]; }
    inline constexpr const T &operator[](int i) const { return m_data[i]; }
    inline constexpr void reset() { m_size = 0; }
    inline constexpr int size() const { return m_size; }
    inline constexpr int capacity() const { return m_capacity; }
    inline constexpr T const *data() const { return m_data; }
    inline constexpr void resize(int size) { m_size = size; }
    inline constexpr void set_data(T *data) { m_data = data; }

  private:
    int m_size;
    int m_capacity;

    T *m_data;
  };

  // ownership of m_data stays within the caller
  template <class T>
  SimpleVector<T> make_SimpleVector(int capacity, T *data) {
    SimpleVector<T> ret;
    ret.construct(capacity, data);
    return ret;
  }

  // ownership of m_data stays within the caller
  template <class T>
  SimpleVector<T> *make_SimpleVector(SimpleVector<T> *mem, int capacity, T *data) {
    auto ret = new (mem) SimpleVector<T>();
    ret->construct(capacity, data);
    return ret;
  }

}  // namespace GPU

namespace GPU {

  template <class T, int maxSize>
  class VecArray {
  public:
    using self = VecArray<T, maxSize>;
    using value_t = T;

    inline constexpr int push_back_unsafe(const T &element) {
      auto previousSize = m_size;
      m_size++;
      if (previousSize < maxSize) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        --m_size;
        return -1;
      }
    }

    template <class... Ts>
    constexpr int emplace_back_unsafe(Ts &&... args) {
      auto previousSize = m_size;
      m_size++;
      if (previousSize < maxSize) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        --m_size;
        return -1;
      }
    }

    inline constexpr T &back() const {
      if (m_size > 0) {
        return m_data[m_size - 1];
      } else
        return T();  //undefined behaviour
    }

    // thread-safe version of the vector, when used in a CUDA kernel
    __device__ int push_back(const T &element) {
      auto previousSize = atomicAdd(&m_size, 1);
      if (previousSize < maxSize) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        atomicSub(&m_size, 1);
        return -1;
      }
    }

    template <class... Ts>
    __device__ int emplace_back(Ts &&... args) {
      auto previousSize = atomicAdd(&m_size, 1);
      if (previousSize < maxSize) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        atomicSub(&m_size, 1);
        return -1;
      }
    }

    inline constexpr T pop_back() {
      if (m_size > 0) {
        auto previousSize = m_size--;
        return m_data[previousSize - 1];
      } else
        return T();
    }

    inline constexpr T const *begin() const { return m_data; }
    inline constexpr T const *end() const { return m_data + m_size; }
    inline constexpr T *begin() { return m_data; }
    inline constexpr T *end() { return m_data + m_size; }
    inline constexpr int size() const { return m_size; }
    inline constexpr T &operator[](int i) { return m_data[i]; }
    inline constexpr const T &operator[](int i) const { return m_data[i]; }
    inline constexpr void reset() { m_size = 0; }
    inline static constexpr int capacity() { return maxSize; }
    inline constexpr T const *data() const { return m_data; }
    inline constexpr void resize(int size) { m_size = size; }
    inline constexpr bool empty() const { return 0 == m_size; }
    inline constexpr bool full() const { return maxSize == m_size; }

  private:
    T m_data[maxSize];

    int m_size;
  };

}  // namespace GPU

class AtomicPairCounter {
public:
  using c_type = unsigned long long int;

  AtomicPairCounter() {}
  AtomicPairCounter(c_type i) { counter.ac = i; }

  __device__ __host__ AtomicPairCounter& operator=(c_type i) {
    counter.ac = i;
    return *this;
  }

  struct Counters {
    uint32_t n;  // in a "One to Many" association is the number of "One"
    uint32_t m;  // in a "One to Many" association is the total number of associations
  };

  union Atomic2 {
    Counters counters;
    c_type ac;
  };

  static constexpr c_type incr = 1UL << 32;

  __device__ __host__ Counters get() const { return counter.counters; }

  // increment n by 1 and m by i.  return previous value
  __host__ __device__ __forceinline__ Counters add(uint32_t i) {
    c_type c = i;
    c += incr;
    Atomic2 ret;
#ifdef __CUDA_ARCH__
    ret.ac = atomicAdd(&counter.ac, c);
#else
    ret.ac = counter.ac;
    counter.ac += c;
#endif
    return ret.counters;
  }

private:
  Atomic2 counter;
};

namespace cuda_std {
  template <typename T = void>
  struct less {
    __host__ __device__ constexpr bool operator()(const T &lhs, const T &rhs) const { return lhs < rhs; }
  };

  template <>
  struct less<void> {
    template <typename T, typename U>
    __host__ __device__ constexpr bool operator()(const T &lhs, const U &rhs) const {
      return lhs < rhs;
    }
  };

  template <typename RandomIt, typename T, typename Compare = less<T>>
  __host__ __device__ constexpr RandomIt upper_bound(RandomIt first, RandomIt last, const T &value, Compare comp = {}) {
    auto count = last - first;

    while (count > 0) {
      auto it = first;
      auto step = count / 2;
      it += step;
      if (!comp(value, *it)) {
        first = ++it;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    return first;
  }
}

namespace cudautils {

  template <typename Histo, typename T>
  __global__ void countFromVector(Histo *__restrict__ h,
                                  uint32_t nh,
                                  T const *__restrict__ v,
                                  uint32_t const *__restrict__ offsets) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = first, nt = offsets[nh]; i < nt; i += gridDim.x * blockDim.x) {
      auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
      assert((*off) > 0);
      int32_t ih = off - offsets - 1;
      assert(ih >= 0);
      assert(ih < int(nh));
      (*h).count(v[i], ih);
    }
  }

  template <typename Histo, typename T>
  __global__ void fillFromVector(Histo *__restrict__ h,
                                 uint32_t nh,
                                 T const *__restrict__ v,
                                 uint32_t const *__restrict__ offsets) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = first, nt = offsets[nh]; i < nt; i += gridDim.x * blockDim.x) {
      auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
      assert((*off) > 0);
      int32_t ih = off - offsets - 1;
      assert(ih >= 0);
      assert(ih < int(nh));
      (*h).fill(v[i], i, ih);
    }
  }

  template <typename Histo>
  inline void launchZero(Histo *__restrict__ h,
                         cudaStream_t stream
  ) {
    uint32_t *off = (uint32_t *)((char *)(h) + offsetof(Histo, off));
    cudaMemsetAsync(off, 0, 4 * Histo::totbins(), stream);
  }

  template <typename Assoc>
  __global__ void finalizeBulk(AtomicPairCounter const *apc, Assoc *__restrict__ assoc) {
    assoc->bulkFinalizeFill(*apc);
  }

}  // namespace cudautils

// iteratate over N bins left and right of the one containing "v"
template <typename Hist, typename V, typename Func>
__host__ __device__ __forceinline__ void forEachInBins(Hist const &hist, V value, int n, Func func) {
  int bs = Hist::bin(value);
  int be = std::min(int(Hist::nbins() - 1), bs + n);
  bs = std::max(0, bs - n);
  assert(be >= bs);
  for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
    func(*pj);
  }
}

// iteratate over bins containing all values in window wmin, wmax
template <typename Hist, typename V, typename Func>
__host__ __device__ __forceinline__ void forEachInWindow(Hist const &hist, V wmin, V wmax, Func const &func) {
  auto bs = Hist::bin(wmin);
  auto be = Hist::bin(wmax);
  assert(be >= bs);
  for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
    func(*pj);
  }
}

template <typename T,                  // the type of the discretized input values
          uint32_t NBINS,              // number of bins
          uint32_t SIZE,               // max number of element
          uint32_t S = sizeof(T) * 8,  // number of significant bits in T
          typename I = uint32_t,  // type stored in the container (usually an index in a vector of the input values)
          uint32_t NHISTS = 1     // number of histos stored
          >
class HistoContainer {
public:
  using Counter = uint32_t;

  using CountersOnly = HistoContainer<T, NBINS, 0, S, I, NHISTS>;

  using index_type = I;
  using UT = typename std::make_unsigned<T>::type;

  static constexpr uint32_t ilog2(uint32_t v) {
    constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
    constexpr uint32_t s[] = {1, 2, 4, 8, 16};

    uint32_t r = 0;  // result of log2(v) will go here
    for (auto i = 4; i >= 0; i--)
      if (v & b[i]) {
        v >>= s[i];
        r |= s[i];
      }
    return r;
  }

  static constexpr uint32_t sizeT() { return S; }
  static constexpr uint32_t nbins() { return NBINS; }
  static constexpr uint32_t nhists() { return NHISTS; }
  static constexpr uint32_t totbins() { return NHISTS * NBINS + 1; }
  static constexpr uint32_t nbits() { return ilog2(NBINS - 1) + 1; }
  static constexpr uint32_t capacity() { return SIZE; }

  static constexpr auto histOff(uint32_t nh) { return NBINS * nh; }

  static constexpr UT bin(T t) {
    constexpr uint32_t shift = sizeT() - nbits();
    constexpr uint32_t mask = (1 << nbits()) - 1;
    return (t >> shift) & mask;
  }

  __host__ __device__ void zero() {
    for (auto &i : off)
      i = 0;
  }

  __host__ __device__ void add(CountersOnly const &co) {
    for (uint32_t i = 0; i < totbins(); ++i) {
#ifdef __CUDA_ARCH__
      atomicAdd(off + i, co.off[i]);
#else
      auto &a = (std::atomic<Counter> &)(off[i]);
      a += co.off[i];
#endif
    }
  }

  static __host__ __device__ __forceinline__ uint32_t atomicIncrement(Counter &x) {
#ifdef __CUDA_ARCH__
    return atomicAdd(&x, 1);
#else
    auto &a = (std::atomic<Counter> &)(x);
    return a++;
#endif
  }

  static __host__ __device__ __forceinline__ uint32_t atomicDecrement(Counter &x) {
#ifdef __CUDA_ARCH__
    return atomicSub(&x, 1);
#else
    auto &a = (std::atomic<Counter> &)(x);
    return a--;
#endif
  }

  __host__ __device__ __forceinline__ void countDirect(T b) {
    assert(b < nbins());
    atomicIncrement(off[b]);
  }

  __host__ __device__ __forceinline__ void fillDirect(T b, index_type j) {
    assert(b < nbins());
    auto w = atomicDecrement(off[b]);
    assert(w > 0);
    bins[w - 1] = j;
  }

  __device__ __host__ __forceinline__ int32_t bulkFill(AtomicPairCounter &apc, index_type const *v, uint32_t n) {
    auto c = apc.add(n);
    if (c.m >= nbins())
      return -int32_t(c.m);
    off[c.m] = c.n;
    for (uint32_t j = 0; j < n; ++j)
      bins[c.n + j] = v[j];
    return c.m;
  }

  __device__ __host__ __forceinline__ void bulkFinalize(AtomicPairCounter const &apc) {
    off[apc.get().m] = apc.get().n;
  }

  __device__ __host__ __forceinline__ void bulkFinalizeFill(AtomicPairCounter const &apc) {
    auto m = apc.get().m;
    auto n = apc.get().n;
    if (m >= nbins()) {  // overflow!
      off[nbins()] = uint32_t(off[nbins() - 1]);
      return;
    }
    auto first = m + blockDim.x * blockIdx.x + threadIdx.x;
    for (auto i = first; i < totbins(); i += gridDim.x * blockDim.x) {
      off[i] = n;
    }
  }

  __host__ __device__ __forceinline__ void count(T t) {
    uint32_t b = bin(t);
    assert(b < nbins());
    atomicIncrement(off[b]);
  }

  __host__ __device__ __forceinline__ void fill(T t, index_type j) {
    uint32_t b = bin(t);
    assert(b < nbins());
    auto w = atomicDecrement(off[b]);
    assert(w > 0);
    bins[w - 1] = j;
  }

  __host__ __device__ __forceinline__ void count(T t, uint32_t nh) {
    uint32_t b = bin(t);
    assert(b < nbins());
    b += histOff(nh);
    assert(b < totbins());
    atomicIncrement(off[b]);
  }

  __host__ __device__ __forceinline__ void fill(T t, index_type j, uint32_t nh) {
    uint32_t b = bin(t);
    assert(b < nbins());
    b += histOff(nh);
    assert(b < totbins());
    auto w = atomicDecrement(off[b]);
    assert(w > 0);
    bins[w - 1] = j;
  }

  __device__ __host__ __forceinline__ void finalize(Counter *ws = nullptr) {
    assert(off[totbins() - 1] == 0);
    blockPrefixScan(off, totbins(), ws);
    assert(off[totbins() - 1] == off[totbins() - 2]);
  }

  constexpr auto size() const { return uint32_t(off[totbins() - 1]); }
  constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

  constexpr index_type const *begin() const { return bins; }
  constexpr index_type const *end() const { return begin() + size(); }

  constexpr index_type const *begin(uint32_t b) const { return bins + off[b]; }
  constexpr index_type const *end(uint32_t b) const { return bins + off[b + 1]; }

  Counter off[totbins()];
  index_type bins[capacity()];
};

template <typename I,        // type stored in the container (usually an index in a vector of the input values)
          uint32_t MAXONES,  // max number of "ones"
          uint32_t MAXMANYS  // max number of "manys"
          >
using OneToManyAssoc = HistoContainer<uint32_t, MAXONES, MAXMANYS, sizeof(uint32_t) * 8, I, 1>;

namespace CAConstants {

  // constants
  constexpr uint32_t maxNumberOfTuples() { return 24 * 1024; }
  constexpr uint32_t maxNumberOfQuadruplets() { return maxNumberOfTuples(); }
  constexpr uint32_t maxNumberOfDoublets() { return 448 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 128; }
  constexpr uint32_t maxNumOfActiveDoublets() { return maxNumberOfDoublets() / 4; }

  constexpr uint32_t maxNumberOfLayerPairs() { return 20; }
  constexpr uint32_t maxNumberOfLayers() { return 10; }
  constexpr uint32_t maxTuples() { return maxNumberOfTuples(); }

  // types
  using hindex_type = uint16_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
  using tindex_type = uint16_t;  //  for tuples

  using CellNeighbors = GPU::VecArray<uint32_t, 36>;
  using CellTracks = GPU::VecArray<tindex_type, 42>;

  using CellNeighborsVector = GPU::SimpleVector<CellNeighbors>;
  using CellTracksVector = GPU::SimpleVector<CellTracks>;

  using OuterHitOfCell = GPU::VecArray<uint32_t, maxCellsPerHit()>;
  using TuplesContainer = OneToManyAssoc<hindex_type, maxTuples(), 5 * maxTuples()>;
  using HitToTuple =
      OneToManyAssoc<tindex_type, pixelGPUConstants::maxNumberOfHits, 4 * maxTuples()>;  // 3.5 should be enough
  using TupleMultiplicity = OneToManyAssoc<tindex_type, 8, maxTuples()>;

}  // namespace CAConstants

template <class T>
class TkRotation;

// to be moved in an external common library???

/** Rotation matrix used by SOA (as in GPU)
 */

template <class T>
class SOARotation {
public:
  constexpr inline SOARotation() {}

  constexpr inline explicit SOARotation(T) : R11(1), R12(0), R13(0), R21(0), R22(1), R23(0), R31(0), R32(0), R33(1) {}

  constexpr inline SOARotation(T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz)
      : R11(xx), R12(xy), R13(xz), R21(yx), R22(yy), R23(yz), R31(zx), R32(zy), R33(zz) {}

  constexpr inline SOARotation(const T *p)
      : R11(p[0]), R12(p[1]), R13(p[2]), R21(p[3]), R22(p[4]), R23(p[5]), R31(p[6]), R32(p[7]), R33(p[8]) {}

  template <typename U>
  constexpr inline SOARotation(const TkRotation<U> &a)
      : R11(a.xx()),
        R12(a.xy()),
        R13(a.xz()),
        R21(a.yx()),
        R22(a.yy()),
        R23(a.yz()),
        R31(a.zx()),
        R32(a.zy()),
        R33(a.zz()) {}

  constexpr inline SOARotation transposed() const { return SOARotation(R11, R21, R31, R12, R22, R32, R13, R23, R33); }

  // if frame this is to local
  constexpr inline void multiply(T const vx, T const vy, T const vz, T &ux, T &uy, T &uz) const {
    ux = R11 * vx + R12 * vy + R13 * vz;
    uy = R21 * vx + R22 * vy + R23 * vz;
    uz = R31 * vx + R32 * vy + R33 * vz;
  }

  // if frame this is to global
  constexpr inline void multiplyInverse(T const vx, T const vy, T const vz, T &ux, T &uy, T &uz) const {
    ux = R11 * vx + R21 * vy + R31 * vz;
    uy = R12 * vx + R22 * vy + R32 * vz;
    uz = R13 * vx + R23 * vy + R33 * vz;
  }

  // if frame this is to global
  constexpr inline void multiplyInverse(T const vx, T const vy, T &ux, T &uy, T &uz) const {
    ux = R11 * vx + R21 * vy;
    uy = R12 * vx + R22 * vy;
    uz = R13 * vx + R23 * vy;
  }

  constexpr inline T const &xx() const { return R11; }
  constexpr inline T const &xy() const { return R12; }
  constexpr inline T const &xz() const { return R13; }
  constexpr inline T const &yx() const { return R21; }
  constexpr inline T const &yy() const { return R22; }
  constexpr inline T const &yz() const { return R23; }
  constexpr inline T const &zx() const { return R31; }
  constexpr inline T const &zy() const { return R32; }
  constexpr inline T const &zz() const { return R33; }

private:
  T R11, R12, R13;
  T R21, R22, R23;
  T R31, R32, R33;
};

template <class T>
class SOAFrame {
public:
  constexpr inline SOAFrame() {}

  constexpr inline SOAFrame(T ix, T iy, T iz, SOARotation<T> const &irot) : px(ix), py(iy), pz(iz), rot(irot) {}

  constexpr inline SOARotation<T> const &rotation() const { return rot; }

  constexpr inline void toLocal(T const vx, T const vy, T const vz, T &ux, T &uy, T &uz) const {
    rot.multiply(vx - px, vy - py, vz - pz, ux, uy, uz);
  }

  constexpr inline void toGlobal(T const vx, T const vy, T const vz, T &ux, T &uy, T &uz) const {
    rot.multiplyInverse(vx, vy, vz, ux, uy, uz);
    ux += px;
    uy += py;
    uz += pz;
  }

  constexpr inline void toGlobal(T const vx, T const vy, T &ux, T &uy, T &uz) const {
    rot.multiplyInverse(vx, vy, ux, uy, uz);
    ux += px;
    uy += py;
    uz += pz;
  }

  constexpr inline void toGlobal(T cxx, T cxy, T cyy, T *gl) const {
    auto const &r = rot;
    gl[0] = r.xx() * (r.xx() * cxx + r.yx() * cxy) + r.yx() * (r.xx() * cxy + r.yx() * cyy);
    gl[1] = r.xx() * (r.xy() * cxx + r.yy() * cxy) + r.yx() * (r.xy() * cxy + r.yy() * cyy);
    gl[2] = r.xy() * (r.xy() * cxx + r.yy() * cxy) + r.yy() * (r.xy() * cxy + r.yy() * cyy);
    gl[3] = r.xx() * (r.xz() * cxx + r.yz() * cxy) + r.yx() * (r.xz() * cxy + r.yz() * cyy);
    gl[4] = r.xy() * (r.xz() * cxx + r.yz() * cxy) + r.yy() * (r.xz() * cxy + r.yz() * cyy);
    gl[5] = r.xz() * (r.xz() * cxx + r.yz() * cxy) + r.yz() * (r.xz() * cxy + r.yz() * cyy);
  }

  constexpr inline void toLocal(T const *ge, T &lxx, T &lxy, T &lyy) const {
    auto const &r = rot;

    T cxx = ge[0];
    T cyx = ge[1];
    T cyy = ge[2];
    T czx = ge[3];
    T czy = ge[4];
    T czz = ge[5];

    lxx = r.xx() * (r.xx() * cxx + r.xy() * cyx + r.xz() * czx) +
          r.xy() * (r.xx() * cyx + r.xy() * cyy + r.xz() * czy) + r.xz() * (r.xx() * czx + r.xy() * czy + r.xz() * czz);
    lxy = r.yx() * (r.xx() * cxx + r.xy() * cyx + r.xz() * czx) +
          r.yy() * (r.xx() * cyx + r.xy() * cyy + r.xz() * czy) + r.yz() * (r.xx() * czx + r.xy() * czy + r.xz() * czz);
    lyy = r.yx() * (r.yx() * cxx + r.yy() * cyx + r.yz() * czx) +
          r.yy() * (r.yx() * cyx + r.yy() * cyy + r.yz() * czy) + r.yz() * (r.yx() * czx + r.yy() * czy + r.yz() * czz);
  }

  constexpr inline T x() const { return px; }
  constexpr inline T y() const { return py; }
  constexpr inline T z() const { return pz; }

private:
  T px, py, pz;
  SOARotation<T> rot;
};

namespace pixelCPEforGPU {

  using Frame = SOAFrame<float>;
  using Rotation = SOARotation<float>;

  // all modules are identical!
  struct CommonParams {
    float theThicknessB;
    float theThicknessE;
    float thePitchX;
    float thePitchY;
  };

  struct DetParams {
    bool isBarrel;
    bool isPosZ;
    uint16_t layer;
    uint16_t index;
    uint32_t rawId;

    float shiftX;
    float shiftY;
    float chargeWidthX;
    float chargeWidthY;

    float x0, y0, z0;  // the vertex in the local coord of the detector

    float sx[3], sy[3];  // the errors...

    Frame frame;
  };

  using phase1PixelTopology::AverageGeometry;

  struct LayerGeometry {
    uint32_t layerStart[phase1PixelTopology::numberOfLayers + 1];
    uint8_t layer[phase1PixelTopology::layerIndexSize];
  };

  struct ParamsOnGPU {
    CommonParams const* m_commonParams;
    DetParams const* m_detParams;
    LayerGeometry const* m_layerGeometry;
    AverageGeometry const* m_averageGeometry;

    constexpr CommonParams const& __restrict__ commonParams() const {
      CommonParams const* __restrict__ l = m_commonParams;
      return *l;
    }
    constexpr DetParams const& __restrict__ detParams(int i) const {
      DetParams const* __restrict__ l = m_detParams;
      return l[i];
    }
    constexpr LayerGeometry const& __restrict__ layerGeometry() const { return *m_layerGeometry; }
    constexpr AverageGeometry const& __restrict__ averageGeometry() const { return *m_averageGeometry; }

    __device__ uint8_t layer(uint16_t id) const {
      return __ldg(m_layerGeometry->layer + id / phase1PixelTopology::maxModuleStride);
    };
  };

  // SOA (on device)
  template <uint32_t N>
  struct ClusParamsT {
    uint32_t minRow[N];
    uint32_t maxRow[N];
    uint32_t minCol[N];
    uint32_t maxCol[N];

    int32_t Q_f_X[N];
    int32_t Q_l_X[N];
    int32_t Q_f_Y[N];
    int32_t Q_l_Y[N];

    int32_t charge[N];

    float xpos[N];
    float ypos[N];

    float xerr[N];
    float yerr[N];

    int16_t xsize[N];  // clipped at 127 if negative is edge....
    int16_t ysize[N];
  };

  constexpr int32_t MaxHitsInIter = gpuClustering::maxHitsInIter();
  using ClusParams = ClusParamsT<MaxHitsInIter>;

  constexpr inline void computeAnglesFromDet(
      DetParams const& __restrict__ detParams, float const x, float const y, float& cotalpha, float& cotbeta) {
    // x,y local position on det
    auto gvx = x - detParams.x0;
    auto gvy = y - detParams.y0;
    auto gvz = -1.f / detParams.z0;
    // normalization not required as only ratio used...
    // calculate angles
    cotalpha = gvx * gvz;
    cotbeta = gvy * gvz;
  }

  constexpr inline float correction(int sizeM1,
                                    int Q_f,                        //!< Charge in the first pixel.
                                    int Q_l,                        //!< Charge in the last pixel.
                                    uint16_t upper_edge_first_pix,  //!< As the name says.
                                    uint16_t lower_edge_last_pix,   //!< As the name says.
                                    float lorentz_shift,            //!< L-shift at half thickness
                                    float theThickness,             //detector thickness
                                    float cot_angle,                //!< cot of alpha_ or beta_
                                    float pitch,                    //!< thePitchX or thePitchY
                                    bool first_is_big,              //!< true if the first is big
                                    bool last_is_big)               //!< true if the last is big
  {
    if (0 == sizeM1)  // size 1
      return 0;

    float W_eff = 0;
    bool simple = true;
    if (1 == sizeM1) {  // size 2
      //--- Width of the clusters minus the edge (first and last) pixels.
      //--- In the note, they are denoted x_F and x_L (and y_F and y_L)
      // assert(lower_edge_last_pix >= upper_edge_first_pix);
      auto W_inner = pitch * float(lower_edge_last_pix - upper_edge_first_pix);  // in cm

      //--- Predicted charge width from geometry
      auto W_pred = theThickness * cot_angle  // geometric correction (in cm)
                    - lorentz_shift;          // (in cm) &&& check fpix!

      W_eff = std::abs(W_pred) - W_inner;

      //--- If the observed charge width is inconsistent with the expectations
      //--- based on the track, do *not* use W_pred-W_inner.  Instead, replace
      //--- it with an *average* effective charge width, which is the average
      //--- length of the edge pixels.
      simple =
          (W_eff < 0.0f) | (W_eff > pitch);  // this produces "large" regressions for very small numeric differences...
    }

    if (simple) {
      //--- Total length of the two edge pixels (first+last)
      float sum_of_edge = 2.0f;
      if (first_is_big)
        sum_of_edge += 1.0f;
      if (last_is_big)
        sum_of_edge += 1.0f;
      W_eff = pitch * 0.5f * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)
    }

    //--- Finally, compute the position in this projection
    float Qdiff = Q_l - Q_f;
    float Qsum = Q_l + Q_f;

    //--- Temporary fix for clusters with both first and last pixel with charge = 0
    if (Qsum == 0)
      Qsum = 1.0f;

    return 0.5f * (Qdiff / Qsum) * W_eff;
  }

  constexpr inline void position(CommonParams const& __restrict__ comParams,
                                 DetParams const& __restrict__ detParams,
                                 ClusParams& cp,
                                 uint32_t ic) {
    //--- Upper Right corner of Lower Left pixel -- in measurement frame
    uint16_t llx = cp.minRow[ic] + 1;
    uint16_t lly = cp.minCol[ic] + 1;

    //--- Lower Left corner of Upper Right pixel -- in measurement frame
    uint16_t urx = cp.maxRow[ic];
    uint16_t ury = cp.maxCol[ic];

    auto llxl = phase1PixelTopology::localX(llx);
    auto llyl = phase1PixelTopology::localY(lly);
    auto urxl = phase1PixelTopology::localX(urx);
    auto uryl = phase1PixelTopology::localY(ury);

    auto mx = llxl + urxl;
    auto my = llyl + uryl;

    auto xsize = int(urxl) + 2 - int(llxl);
    auto ysize = int(uryl) + 2 - int(llyl);
    assert(xsize >= 0);  // 0 if bixpix...
    assert(ysize >= 0);

    if (phase1PixelTopology::isBigPixX(cp.minRow[ic]))
      ++xsize;
    if (phase1PixelTopology::isBigPixX(cp.maxRow[ic]))
      ++xsize;
    if (phase1PixelTopology::isBigPixY(cp.minCol[ic]))
      ++ysize;
    if (phase1PixelTopology::isBigPixY(cp.maxCol[ic]))
      ++ysize;

    int unbalanceX = 8. * std::abs(float(cp.Q_f_X[ic] - cp.Q_l_X[ic])) / float(cp.Q_f_X[ic] + cp.Q_l_X[ic]);
    int unbalanceY = 8. * std::abs(float(cp.Q_f_Y[ic] - cp.Q_l_Y[ic])) / float(cp.Q_f_Y[ic] + cp.Q_l_Y[ic]);
    xsize = 8 * xsize - unbalanceX;
    ysize = 8 * ysize - unbalanceY;

    cp.xsize[ic] = std::min(xsize, 1023);
    cp.ysize[ic] = std::min(ysize, 1023);

    if (cp.minRow[ic] == 0 || cp.maxRow[ic] == phase1PixelTopology::lastRowInModule)
      cp.xsize[ic] = -cp.xsize[ic];
    if (cp.minCol[ic] == 0 || cp.maxCol[ic] == phase1PixelTopology::lastColInModule)
      cp.ysize[ic] = -cp.ysize[ic];

    // apply the lorentz offset correction
    auto xPos = detParams.shiftX + comParams.thePitchX * (0.5f * float(mx) + float(phase1PixelTopology::xOffset));
    auto yPos = detParams.shiftY + comParams.thePitchY * (0.5f * float(my) + float(phase1PixelTopology::yOffset));

    float cotalpha = 0, cotbeta = 0;

    computeAnglesFromDet(detParams, xPos, yPos, cotalpha, cotbeta);

    auto thickness = detParams.isBarrel ? comParams.theThicknessB : comParams.theThicknessE;

    auto xcorr = correction(cp.maxRow[ic] - cp.minRow[ic],
                            cp.Q_f_X[ic],
                            cp.Q_l_X[ic],
                            llxl,
                            urxl,
                            detParams.chargeWidthX,  // lorentz shift in cm
                            thickness,
                            cotalpha,
                            comParams.thePitchX,
                            phase1PixelTopology::isBigPixX(cp.minRow[ic]),
                            phase1PixelTopology::isBigPixX(cp.maxRow[ic]));

    auto ycorr = correction(cp.maxCol[ic] - cp.minCol[ic],
                            cp.Q_f_Y[ic],
                            cp.Q_l_Y[ic],
                            llyl,
                            uryl,
                            detParams.chargeWidthY,  // lorentz shift in cm
                            thickness,
                            cotbeta,
                            comParams.thePitchY,
                            phase1PixelTopology::isBigPixY(cp.minCol[ic]),
                            phase1PixelTopology::isBigPixY(cp.maxCol[ic]));

    cp.xpos[ic] = xPos + xcorr;
    cp.ypos[ic] = yPos + ycorr;
  }

#ifdef FOO_CPP17
  constexpr inline void errorFromSize(CommonParams const& __restrict__ comParams,
                                      DetParams const& __restrict__ detParams,
                                      ClusParams& cp,
                                      uint32_t ic) {
    // Edge cluster errors
    cp.xerr[ic] = 0.0050;
    cp.yerr[ic] = 0.0085;

    // FIXME these are errors form Run1
    constexpr float xerr_barrel_l1[] = {0.00115, 0.00120, 0.00088};
    constexpr float xerr_barrel_l1_def = 0.00200;  // 0.01030;
    constexpr float yerr_barrel_l1[] = {
        0.00375, 0.00230, 0.00250, 0.00250, 0.00230, 0.00230, 0.00210, 0.00210, 0.00240};
    constexpr float yerr_barrel_l1_def = 0.00210;
    constexpr float xerr_barrel_ln[] = {0.00115, 0.00120, 0.00088};
    constexpr float xerr_barrel_ln_def = 0.00200;  // 0.01030;
    constexpr float yerr_barrel_ln[] = {
        0.00375, 0.00230, 0.00250, 0.00250, 0.00230, 0.00230, 0.00210, 0.00210, 0.00240};
    constexpr float yerr_barrel_ln_def = 0.00210;
    constexpr float xerr_endcap[] = {0.0020, 0.0020};
    constexpr float xerr_endcap_def = 0.0020;
    constexpr float yerr_endcap[] = {0.00210};
    constexpr float yerr_endcap_def = 0.00210;

    auto sx = cp.maxRow[ic] - cp.minRow[ic];
    auto sy = cp.maxCol[ic] - cp.minCol[ic];

    // is edgy ?
    bool isEdgeX = cp.minRow[ic] == 0 or cp.maxRow[ic] == phase1PixelTopology::lastRowInModule;
    bool isEdgeY = cp.minCol[ic] == 0 or cp.maxCol[ic] == phase1PixelTopology::lastColInModule;
    // is one and big?
    bool isBig1X = (0 == sx) && phase1PixelTopology::isBigPixX(cp.minRow[ic]);
    bool isBig1Y = (0 == sy) && phase1PixelTopology::isBigPixY(cp.minCol[ic]);

    if (!isEdgeX && !isBig1X) {
      if (not detParams.isBarrel) {
        cp.xerr[ic] = sx < std::size(xerr_endcap) ? xerr_endcap[sx] : xerr_endcap_def;
      } else if (detParams.layer == 1) {
        cp.xerr[ic] = sx < std::size(xerr_barrel_l1) ? xerr_barrel_l1[sx] : xerr_barrel_l1_def;
      } else {
        cp.xerr[ic] = sx < std::size(xerr_barrel_ln) ? xerr_barrel_ln[sx] : xerr_barrel_ln_def;
      }
    }

    if (!isEdgeY && !isBig1Y) {
      if (not detParams.isBarrel) {
        cp.yerr[ic] = sy < std::size(yerr_endcap) ? yerr_endcap[sy] : yerr_endcap_def;
      } else if (detParams.layer == 1) {
        cp.yerr[ic] = sy < std::size(yerr_barrel_l1) ? yerr_barrel_l1[sy] : yerr_barrel_l1_def;
      } else {
        cp.yerr[ic] = sy < std::size(yerr_barrel_ln) ? yerr_barrel_ln[sy] : yerr_barrel_ln_def;
      }
    }
  }
#endif

  constexpr inline void errorFromDB(CommonParams const& __restrict__ comParams,
                                    DetParams const& __restrict__ detParams,
                                    ClusParams& cp,
                                    uint32_t ic) {
    // Edge cluster errors
    cp.xerr[ic] = 0.0050f;
    cp.yerr[ic] = 0.0085f;

    auto sx = cp.maxRow[ic] - cp.minRow[ic];
    auto sy = cp.maxCol[ic] - cp.minCol[ic];

    // is edgy ?
    bool isEdgeX = cp.minRow[ic] == 0 or cp.maxRow[ic] == phase1PixelTopology::lastRowInModule;
    bool isEdgeY = cp.minCol[ic] == 0 or cp.maxCol[ic] == phase1PixelTopology::lastColInModule;
    // is one and big?
    uint32_t ix = (0 == sx);
    uint32_t iy = (0 == sy);
    ix += (0 == sx) && phase1PixelTopology::isBigPixX(cp.minRow[ic]);
    iy += (0 == sy) && phase1PixelTopology::isBigPixY(cp.minCol[ic]);

    if (not isEdgeX)
      cp.xerr[ic] = detParams.sx[ix];
    if (not isEdgeY)
      cp.yerr[ic] = detParams.sy[iy];
  }
}  // namespace pixelCPEforGPU


class TrackingRecHit2DSOAView {
public:
  static constexpr uint32_t maxHits() { return gpuClustering::MaxNumClusters; }
  using hindex_type = uint16_t;  // if above is <=2^16

  using Hist = HistoContainer<int16_t, 128, gpuClustering::MaxNumClusters, 8 * sizeof(int16_t), uint16_t, 10>;

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  template <typename>
  friend class TrackingRecHit2DHeterogeneous;

  __device__ __forceinline__ uint32_t nHits() const { return m_nHits; }

  __device__ __forceinline__ float& xLocal(int i) { return m_xl[i]; }
  __device__ __forceinline__ float xLocal(int i) const { return __ldg(m_xl + i); }
  __device__ __forceinline__ float& yLocal(int i) { return m_yl[i]; }
  __device__ __forceinline__ float yLocal(int i) const { return __ldg(m_yl + i); }

  __device__ __forceinline__ float& xerrLocal(int i) { return m_xerr[i]; }
  __device__ __forceinline__ float xerrLocal(int i) const { return __ldg(m_xerr + i); }
  __device__ __forceinline__ float& yerrLocal(int i) { return m_yerr[i]; }
  __device__ __forceinline__ float yerrLocal(int i) const { return __ldg(m_yerr + i); }

  __device__ __forceinline__ float& xGlobal(int i) { return m_xg[i]; }
  __device__ __forceinline__ float xGlobal(int i) const { return __ldg(m_xg + i); }
  __device__ __forceinline__ float& yGlobal(int i) { return m_yg[i]; }
  __device__ __forceinline__ float yGlobal(int i) const { return __ldg(m_yg + i); }
  __device__ __forceinline__ float& zGlobal(int i) { return m_zg[i]; }
  __device__ __forceinline__ float zGlobal(int i) const { return __ldg(m_zg + i); }
  __device__ __forceinline__ float& rGlobal(int i) { return m_rg[i]; }
  __device__ __forceinline__ float rGlobal(int i) const { return __ldg(m_rg + i); }

  __device__ __forceinline__ int16_t& iphi(int i) { return m_iphi[i]; }
  __device__ __forceinline__ int16_t iphi(int i) const { return __ldg(m_iphi + i); }

  __device__ __forceinline__ int32_t& charge(int i) { return m_charge[i]; }
  __device__ __forceinline__ int32_t charge(int i) const { return __ldg(m_charge + i); }
  __device__ __forceinline__ int16_t& clusterSizeX(int i) { return m_xsize[i]; }
  __device__ __forceinline__ int16_t clusterSizeX(int i) const { return __ldg(m_xsize + i); }
  __device__ __forceinline__ int16_t& clusterSizeY(int i) { return m_ysize[i]; }
  __device__ __forceinline__ int16_t clusterSizeY(int i) const { return __ldg(m_ysize + i); }
  __device__ __forceinline__ uint16_t& detectorIndex(int i) { return m_detInd[i]; }
  __device__ __forceinline__ uint16_t detectorIndex(int i) const { return __ldg(m_detInd + i); }

  //__device__ __forceinline__ pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }

  //__device__ __forceinline__ uint32_t hitsModuleStart(int i) const { return __ldg(m_hitsModuleStart + i); }

  __device__ __forceinline__ uint32_t* hitsLayerStart() { return m_hitsLayerStart; }
  __device__ __forceinline__ uint32_t const* hitsLayerStart() const { return m_hitsLayerStart; }

  __device__ __forceinline__ Hist& phiBinner() { return *m_hist; }
  __device__ __forceinline__ Hist const& phiBinner() const { return *m_hist; }

  __device__ __forceinline__ AverageGeometry& averageGeometry() { return *m_averageGeometry; }
  __device__ __forceinline__ AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

  //private:
  // local coord
  float *m_xl, *m_yl;
  float *m_xerr, *m_yerr;

  // global coord
  float *m_xg, *m_yg, *m_zg, *m_rg;
  int16_t* m_iphi;

  // cluster properties
  int32_t* m_charge;
  int16_t* m_xsize;
  int16_t* m_ysize;
  uint16_t* m_detInd;

  // supporting objects
  AverageGeometry* m_averageGeometry;  // owned (corrected for beam spot: not sure where to host it otherwise)
  //pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
  //uint32_t const* m_hitsModuleStart;               // forwarded from clusters

  uint32_t* m_hitsLayerStart;

  Hist* m_hist;

  uint32_t m_nHits;
};

namespace trackQuality {
  enum Quality : uint8_t { bad = 0, dup, loose, strict, tight, highPurity };
}

namespace pixelTrack {
  constexpr uint32_t maxNumber() { return 2 * 1024; }

  using hindex_type = uint16_t;
  using HitContainer = OneToManyAssoc<hindex_type, maxNumber(), maxNumber() * maxNumber()>;
  using Quality = trackQuality::Quality;

}  // namespace pixelTrack

template <typename T>
class CircleEq {
public:
  CircleEq() {}

  constexpr CircleEq(T x1, T y1, T x2, T y2, T x3, T y3) { compute(x1, y1, x2, y2, x3, y3); }

  constexpr void compute(T x1, T y1, T x2, T y2, T x3, T y3);

  // dca to origin divided by curvature
  constexpr T dca0() const {
    auto x = m_c * m_xp + m_alpha;
    auto y = m_c * m_yp + m_beta;
    return std::sqrt(x * x + y * y) - T(1);
  }

  // dca to given point (divided by curvature)
  constexpr T dca(T x, T y) const {
    x = m_c * (m_xp - x) + m_alpha;
    y = m_c * (m_yp - y) + m_beta;
    return std::sqrt(x * x + y * y) - T(1);
  }

  // curvature
  constexpr auto curvature() const { return m_c; }

  // alpha and beta
  constexpr std::pair<T, T> cosdir() const { return std::make_pair(m_alpha, m_beta); }

  // alpha and beta af given point
  constexpr std::pair<T, T> cosdir(T x, T y) const {
    return std::make_pair(m_alpha - m_c * (x - m_xp), m_beta - m_c * (y - m_yp));
  }

  // center
  constexpr std::pair<T, T> center() const { return std::make_pair(m_xp + m_alpha / m_c, m_yp + m_beta / m_c); }

  constexpr auto radius() const { return T(1) / m_c; }

  T m_xp = 0;
  T m_yp = 0;
  T m_c = 0;
  T m_alpha = 0;
  T m_beta = 0;
};

template <typename T>
constexpr void CircleEq<T>::compute(T x1, T y1, T x2, T y2, T x3, T y3) {
  bool noflip = std::abs(x3 - x1) < std::abs(y3 - y1);

  auto x1p = noflip ? x1 - x2 : y1 - y2;
  auto y1p = noflip ? y1 - y2 : x1 - x2;
  auto d12 = x1p * x1p + y1p * y1p;
  auto x3p = noflip ? x3 - x2 : y3 - y2;
  auto y3p = noflip ? y3 - y2 : x3 - x2;
  auto d32 = x3p * x3p + y3p * y3p;

  auto num = x1p * y3p - y1p * x3p;  // num also gives correct sign for CT
  auto det = d12 * y3p - d32 * y1p;

  /*
  auto ct  = num/det;
  auto sn  = det>0 ? T(1.) : T(-1.);
  auto st2 = (d12*x3p-d32*x1p)/det;
  auto seq = T(1.) +st2*st2;
  auto al2 = sn/std::sqrt(seq);
  auto be2 = -st2*al2;
  ct *= T(2.)*al2;
  */

  auto st2 = (d12 * x3p - d32 * x1p);
  auto seq = det * det + st2 * st2;
  auto al2 = T(1.) / std::sqrt(seq);
  auto be2 = -st2 * al2;
  auto ct = T(2.) * num * al2;
  al2 *= det;

  m_xp = x2;
  m_yp = y2;
  m_c = noflip ? ct : -ct;
  m_alpha = noflip ? al2 : -be2;
  m_beta = noflip ? be2 : -al2;
}

class GPUCACell {
public:
  using ptrAsInt = unsigned long long;

  static constexpr int maxCellsPerHit = CAConstants::maxCellsPerHit();
  using OuterHitOfCell = CAConstants::OuterHitOfCell;
  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  using Hits = TrackingRecHit2DSOAView;
  using hindex_type = Hits::hindex_type;

  using TmpTuple = GPU::VecArray<uint32_t, 6>;

  using HitContainer = pixelTrack::HitContainer;
  using Quality = trackQuality::Quality;
  static constexpr auto bad = trackQuality::bad;

  GPUCACell() = default;

  __device__ __forceinline__ void init(/*CellNeighborsVector& cellNeighbors,
                                         CellTracksVector& cellTracks,*/
                                       Hits const& hh,
                                       int layerPairId,
                                       int doubletId,
                                       hindex_type innerHitId,
                                       hindex_type outerHitId) {
    theInnerHitId = innerHitId;
    theOuterHitId = outerHitId;
    theDoubletId = doubletId;
    theLayerPairId = layerPairId;
    theUsed = 0;

    // optimization that depends on access pattern
    theInnerZ = hh.zGlobal(innerHitId);
    theInnerR = hh.rGlobal(innerHitId);

    outerNeighbors().reset();
    tracks().reset();
    assert(outerNeighbors().empty());
    assert(tracks().empty());
  }

  __device__ __forceinline__ int addOuterNeighbor(CellNeighbors::value_t t, CellNeighborsVector& cellNeighbors) {
    return outerNeighbors().push_back(t);
  }

  __device__ __forceinline__ int addTrack(CellTracks::value_t t, CellTracksVector& cellTracks) {
    return tracks().push_back(t);
  }

  __device__ __forceinline__ CellTracks& tracks() { return theTracks; }
  __device__ __forceinline__ CellTracks const& tracks() const { return theTracks; }
  __device__ __forceinline__ CellNeighbors& outerNeighbors() { return theOuterNeighbors; }
  __device__ __forceinline__ CellNeighbors const& outerNeighbors() const { return theOuterNeighbors; }
  __device__ __forceinline__ float get_inner_x(Hits const& hh) const { return hh.xGlobal(theInnerHitId); }
  __device__ __forceinline__ float get_outer_x(Hits const& hh) const { return hh.xGlobal(theOuterHitId); }
  __device__ __forceinline__ float get_inner_y(Hits const& hh) const { return hh.yGlobal(theInnerHitId); }
  __device__ __forceinline__ float get_outer_y(Hits const& hh) const { return hh.yGlobal(theOuterHitId); }
  __device__ __forceinline__ float get_inner_z(Hits const& hh) const { return theInnerZ; }
  // { return hh.zGlobal(theInnerHitId); } // { return theInnerZ; }
  __device__ __forceinline__ float get_outer_z(Hits const& hh) const { return hh.zGlobal(theOuterHitId); }
  __device__ __forceinline__ float get_inner_r(Hits const& hh) const { return theInnerR; }
  // { return hh.rGlobal(theInnerHitId); } // { return theInnerR; }
  __device__ __forceinline__ float get_outer_r(Hits const& hh) const { return hh.rGlobal(theOuterHitId); }

  __device__ __forceinline__ auto get_inner_iphi(Hits const& hh) const { return hh.iphi(theInnerHitId); }
  __device__ __forceinline__ auto get_outer_iphi(Hits const& hh) const { return hh.iphi(theOuterHitId); }

  __device__ __forceinline__ float get_inner_detIndex(Hits const& hh) const { return hh.detectorIndex(theInnerHitId); }
  __device__ __forceinline__ float get_outer_detIndex(Hits const& hh) const { return hh.detectorIndex(theOuterHitId); }

  constexpr unsigned int get_inner_hit_id() const { return theInnerHitId; }
  constexpr unsigned int get_outer_hit_id() const { return theOuterHitId; }

  __device__ void print_cell() const {
    printf("printing cell: %d, on layerPair: %d, innerHitId: %d, outerHitId: %d \n",
           theDoubletId,
           theLayerPairId,
           theInnerHitId,
           theOuterHitId);
  }

  __device__ bool check_alignment(Hits const& hh,
                                  GPUCACell const& otherCell,
                                  const float ptmin,
                                  const float hardCurvCut,
                                  const float CAThetaCutBarrel,
                                  const float CAThetaCutForward,
                                  const float dcaCutInnerTriplet,
                                  const float dcaCutOuterTriplet) const {
    // detIndex of the layerStart for the Phase1 Pixel Detector:
    // [BPX1, BPX2, BPX3, BPX4,  FP1,  FP2,  FP3,  FN1,  FN2,  FN3, LAST_VALID]
    // [   0,   96,  320,  672, 1184, 1296, 1408, 1520, 1632, 1744,       1856]
    constexpr uint32_t last_bpix1_detIndex = 96;
    constexpr uint32_t last_barrel_detIndex = 1184;
    auto ri = get_inner_r(hh);
    auto zi = get_inner_z(hh);

    auto ro = get_outer_r(hh);
    auto zo = get_outer_z(hh);

    auto r1 = otherCell.get_inner_r(hh);
    auto z1 = otherCell.get_inner_z(hh);
    auto isBarrel = otherCell.get_outer_detIndex(hh) < last_barrel_detIndex;
    bool aligned = areAlignedRZ(r1,
                                z1,
                                ri,
                                zi,
                                ro,
                                zo,
                                ptmin,
                                isBarrel ? CAThetaCutBarrel : CAThetaCutForward);  // 2.f*thetaCut); // FIXME tune cuts
    return (aligned &&
            dcaCut(hh,
                   otherCell,
                   otherCell.get_inner_detIndex(hh) < last_bpix1_detIndex ? dcaCutInnerTriplet : dcaCutOuterTriplet,
                   hardCurvCut));  // FIXME tune cuts
  }

  __device__ __forceinline__ static bool areAlignedRZ(
      float r1, float z1, float ri, float zi, float ro, float zo, const float ptmin, const float thetaCut) {
    float radius_diff = std::abs(r1 - ro);
    float distance_13_squared = radius_diff * radius_diff + (z1 - zo) * (z1 - zo);

    float pMin = ptmin * std::sqrt(distance_13_squared);  // this needs to be divided by
                                                          // radius_diff later

    float tan_12_13_half_mul_distance_13_squared = fabs(z1 * (ri - ro) + zi * (ro - r1) + zo * (r1 - ri));
    return tan_12_13_half_mul_distance_13_squared * pMin <= thetaCut * distance_13_squared * radius_diff;
  }

  __device__ inline bool dcaCut(Hits const& hh,
                                GPUCACell const& otherCell,
                                const float region_origin_radius_plus_tolerance,
                                const float maxCurv) const {
    auto x1 = otherCell.get_inner_x(hh);
    auto y1 = otherCell.get_inner_y(hh);

    auto x2 = get_inner_x(hh);
    auto y2 = get_inner_y(hh);

    auto x3 = get_outer_x(hh);
    auto y3 = get_outer_y(hh);

    CircleEq<float> eq(x1, y1, x2, y2, x3, y3);

    if (eq.curvature() > maxCurv)
      return false;

    return std::abs(eq.dca0()) < region_origin_radius_plus_tolerance * std::abs(eq.curvature());
  }

  __device__ __forceinline__ static bool dcaCutH(float x1,
                                                 float y1,
                                                 float x2,
                                                 float y2,
                                                 float x3,
                                                 float y3,
                                                 const float region_origin_radius_plus_tolerance,
                                                 const float maxCurv) {
    CircleEq<float> eq(x1, y1, x2, y2, x3, y3);

    if (eq.curvature() > maxCurv)
      return false;

    return std::abs(eq.dca0()) < region_origin_radius_plus_tolerance * std::abs(eq.curvature());
  }

  __device__ inline bool hole0(Hits const& hh, GPUCACell const& innerCell) const {
    constexpr uint32_t max_ladder_bpx0 = 12;
    constexpr uint32_t first_ladder_bpx0 = 0;
    constexpr float module_length = 6.7f;
    constexpr float module_tolerance = 0.4f;  // projection to cylinder is inaccurate on BPIX1
    int p = innerCell.get_inner_iphi(hh);
    if (p < 0)
      p += std::numeric_limits<unsigned short>::max();
    p = (max_ladder_bpx0 * p) / std::numeric_limits<unsigned short>::max();
    p %= max_ladder_bpx0;
    auto il = first_ladder_bpx0 + p;
    auto r0 = hh.averageGeometry().ladderR[il];
    auto ri = innerCell.get_inner_r(hh);
    auto zi = innerCell.get_inner_z(hh);
    auto ro = get_outer_r(hh);
    auto zo = get_outer_z(hh);
    auto z0 = zi + (r0 - ri) * (zo - zi) / (ro - ri);
    auto z_in_ladder = std::abs(z0 - hh.averageGeometry().ladderZ[il]);
    auto z_in_module = z_in_ladder - module_length * int(z_in_ladder / module_length);
    auto gap = z_in_module < module_tolerance || z_in_module > (module_length - module_tolerance);
    return gap;
  }

  __device__ inline bool hole4(Hits const& hh, GPUCACell const& innerCell) const {
    constexpr uint32_t max_ladder_bpx4 = 64;
    constexpr uint32_t first_ladder_bpx4 = 84;
    // constexpr float radius_even_ladder = 15.815f;
    // constexpr float radius_odd_ladder = 16.146f;
    constexpr float module_length = 6.7f;
    constexpr float module_tolerance = 0.2f;
    // constexpr float barrel_z_length = 26.f;
    // constexpr float forward_z_begin = 32.f;
    int p = get_outer_iphi(hh);
    if (p < 0)
      p += std::numeric_limits<unsigned short>::max();
    p = (max_ladder_bpx4 * p) / std::numeric_limits<unsigned short>::max();
    p %= max_ladder_bpx4;
    auto il = first_ladder_bpx4 + p;
    auto r4 = hh.averageGeometry().ladderR[il];
    auto ri = innerCell.get_inner_r(hh);
    auto zi = innerCell.get_inner_z(hh);
    auto ro = get_outer_r(hh);
    auto zo = get_outer_z(hh);
    auto z4 = zo + (r4 - ro) * (zo - zi) / (ro - ri);
    auto z_in_ladder = std::abs(z4 - hh.averageGeometry().ladderZ[il]);
    auto z_in_module = z_in_ladder - module_length * int(z_in_ladder / module_length);
    auto gap = z_in_module < module_tolerance || z_in_module > (module_length - module_tolerance);
    auto holeP = z4 > hh.averageGeometry().ladderMaxZ[il] && z4 < hh.averageGeometry().endCapZ[0];
    auto holeN = z4 < hh.averageGeometry().ladderMinZ[il] && z4 > hh.averageGeometry().endCapZ[1];
    return gap || holeP || holeN;
  }

  // trying to free the track building process from hardcoded layers, leaving
  // the visit of the graph based on the neighborhood connections between cells.
  __device__ inline void find_ntuplets(Hits const& hh,
                                       GPUCACell* __restrict__ cells,
                                       CellTracksVector& cellTracks,
                                       HitContainer& foundNtuplets,
                                       AtomicPairCounter& apc,
                                       Quality* __restrict__ quality,
                                       TmpTuple& tmpNtuplet,
                                       const unsigned int minHitsPerNtuplet,
                                       bool startAt0) const {
    // the building process for a track ends if:
    // it has no right neighbor
    // it has no compatible neighbor
    // the ntuplets is then saved if the number of hits it contains is greater
    // than a threshold

    tmpNtuplet.push_back_unsafe(theDoubletId);
    assert(tmpNtuplet.size() <= 4);

    bool last = true;
    for (int j = 0; j < outerNeighbors().size(); ++j) {
      auto otherCell = outerNeighbors()[j];
      if (cells[otherCell].theDoubletId < 0)
        continue;  // killed by earlyFishbone
      last = false;
      cells[otherCell].find_ntuplets(
          hh, cells, cellTracks, foundNtuplets, apc, quality, tmpNtuplet, minHitsPerNtuplet, startAt0);
    }
    if (last) {  // if long enough save...
      if ((unsigned int)(tmpNtuplet.size()) >= minHitsPerNtuplet - 1) {
#ifdef ONLY_TRIPLETS_IN_HOLE
        // triplets accepted only pointing to the hole
        if (tmpNtuplet.size() >= 3 || (startAt0 && hole4(hh, cells[tmpNtuplet[0]])) ||
            ((!startAt0) && hole0(hh, cells[tmpNtuplet[0]])))
#endif
        {
          hindex_type hits[6];
          auto nh = 0U;
          for (auto c : tmpNtuplet) {
            hits[nh++] = cells[c].theInnerHitId;
          }
          hits[nh] = theOuterHitId;
          auto it = foundNtuplets.bulkFill(apc, hits, tmpNtuplet.size() + 1);
          if (it >= 0) {  // if negative is overflow....
            for (auto c : tmpNtuplet)
              cells[c].addTrack(it, cellTracks);
            quality[it] = bad;  // initialize to bad
          }
        }
      }
    }
    tmpNtuplet.pop_back();
    assert(tmpNtuplet.size() < 4);
  }

private:
  CellNeighbors theOuterNeighbors;
  CellTracks theTracks;

public:
  int32_t theDoubletId;
  int16_t theLayerPairId;
  uint16_t theUsed;  // tbd

private:
  float theInnerZ;
  float theInnerR;
  hindex_type theInnerHitId;
  hindex_type theOuterHitId;
};

constexpr float short2phi(short x) {
  constexpr float i2p = M_PI / ((int)(std::numeric_limits<short>::max()) + 1);
  return float(x) * i2p;
}

namespace gpuPixelDoubletsAlgos {

  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  __device__ __forceinline__ void doubletsFromHisto(uint8_t const* __restrict__ layerPairs,
                                                    uint32_t nPairs,
                                                    GPUCACell* cells,
                                                    uint32_t* nCells,
                                                    //CellNeighborsVector* cellNeighbors,
                                                    //CellTracksVector* cellTracks,
                                                    TrackingRecHit2DSOAView const& __restrict__ hh,
                                                    GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                                    int16_t const* __restrict__ phicuts,
                                                    float const* __restrict__ minz,
                                                    float const* __restrict__ maxz,
                                                    float const* __restrict__ maxr,
                                                    bool ideal_cond,
                                                    bool doClusterCut,
                                                    bool doZCut,
                                                    bool doPhiCut,
                                                    uint32_t maxNumOfDoublets) {
    // ysize cuts (z in the barrel)  times 8
    // these are used if doClusterCut is true
    constexpr int minYsizeB1 = 36;
    constexpr int minYsizeB2 = 28;
    constexpr int maxDYsize12 = 28;
    constexpr int maxDYsize = 20;
    constexpr int maxDYPred = 20;
    constexpr float dzdrFact = 8 * 0.0285 / 0.015;  // from dz/dr to "DY"

    bool isOuterLadder = ideal_cond;

    using Hist = TrackingRecHit2DSOAView::Hist;

    auto const& __restrict__ hist = hh.phiBinner();
    uint32_t const* __restrict__ offsets = hh.hitsLayerStart();
    assert(offsets);

    auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };

    // nPairsMax to be optimized later (originally was 64).
    // If it should be much bigger, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    const int nPairsMax = CAConstants::maxNumberOfLayerPairs();
    assert(nPairs <= nPairsMax);
    __shared__ uint32_t innerLayerCumulativeSize[nPairsMax];
    __shared__ uint32_t ntot;
    if (threadIdx.y == 0 && threadIdx.x == 0) {
      innerLayerCumulativeSize[0] = layerSize(layerPairs[0]);
      for (uint32_t i = 1; i < nPairs; ++i) {
        innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i - 1] + layerSize(layerPairs[2 * i]);
      }
      ntot = innerLayerCumulativeSize[nPairs - 1];
    }
    __syncthreads();

    // x runs faster
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    auto first = threadIdx.x;
    auto stride = blockDim.x;

    uint32_t pairLayerId = 0;  // cannot go backward
    for (auto j = idy; j < ntot; j += blockDim.y * gridDim.y) {
      while (j >= innerLayerCumulativeSize[pairLayerId++])
        ;
      --pairLayerId;  // move to lower_bound ??

      assert(pairLayerId < nPairs);
      assert(j < innerLayerCumulativeSize[pairLayerId]);
      assert(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId - 1]);

      uint8_t inner = layerPairs[2 * pairLayerId];
      uint8_t outer = layerPairs[2 * pairLayerId + 1];
      assert(outer > inner);

      auto hoff = Hist::histOff(outer);

      auto i = (0 == pairLayerId) ? j : j - innerLayerCumulativeSize[pairLayerId - 1];
      i += offsets[inner];

      // printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

      assert(i >= offsets[inner]);
      assert(i < offsets[inner + 1]);

      // found hit corresponding to our cuda thread, now do the job
      auto mi = hh.detectorIndex(i);
      if (mi > 2000)
        continue;  // invalid

      auto mez = hh.zGlobal(i);

      if (doZCut && (mez < minz[pairLayerId] || mez > maxz[pairLayerId]))
        continue;

      int16_t mes = -1;  // make compiler happy
      if (doClusterCut) {
        // if ideal treat inner ladder as outer
        if (inner == 0)
          assert(mi < 96);
        isOuterLadder = ideal_cond ? true : 0 == (mi / 8) % 2;  // only for B1/B2/B3 B4 is opposite, FPIX:noclue...

        // in any case we always test mes>0 ...
        mes = inner > 0 || isOuterLadder ? hh.clusterSizeY(i) : -1;

        if (inner == 0 && outer > 3)  // B1 and F1
          if (mes > 0 && mes < minYsizeB1)
            continue;                 // only long cluster  (5*8)
        if (inner == 1 && outer > 3)  // B2 and F1
          if (mes > 0 && mes < minYsizeB2)
            continue;
      }
      auto mep = hh.iphi(i);
      auto mer = hh.rGlobal(i);

      // all cuts: true if fails
      constexpr float z0cut = 12.f;      // cm
      constexpr float hardPtCut = 0.5f;  // GeV
      constexpr float minRadius =
          hardPtCut * 87.78f;  // cm (1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field)
      constexpr float minRadius2T4 = 4.f * minRadius * minRadius;
      auto ptcut = [&](int j, int16_t mop) {
        auto r2t4 = minRadius2T4;
        auto ri = mer;
        auto ro = hh.rGlobal(j);
        // auto mop = hh.iphi(j);
        auto dphi = short2phi(std::min(std::abs(int16_t(mep - mop)), std::abs(int16_t(mop - mep))));
        return dphi * dphi * (r2t4 - ri * ro) > (ro - ri) * (ro - ri);
      };
      auto z0cutoff = [&](int j) {
        auto zo = hh.zGlobal(j);
        auto ro = hh.rGlobal(j);
        auto dr = ro - mer;
        return dr > maxr[pairLayerId] || dr < 0 || std::abs((mez * ro - mer * zo)) > z0cut * dr;
      };

      auto zsizeCut = [&](int j) {
        auto onlyBarrel = outer < 4;
        auto so = hh.clusterSizeY(j);
        auto dy = inner == 0 ? maxDYsize12 : maxDYsize;
        // in the barrel cut on difference in size
        // in the endcap on the prediction on the first layer (actually in the barrel only: happen to be safe for endcap as well)
        // FIXME move pred cut to z0cutoff to optmize loading of and computaiton ...
        auto zo = hh.zGlobal(j);
        auto ro = hh.rGlobal(j);
        return onlyBarrel ? mes > 0 && so > 0 && std::abs(so - mes) > dy
                          : (inner < 4) && mes > 0 &&
                                std::abs(mes - int(std::abs((mez - zo) / (mer - ro)) * dzdrFact + 0.5f)) > maxDYPred;
      };

      auto iphicut = phicuts[pairLayerId];

      auto kl = Hist::bin(int16_t(mep - iphicut));
      auto kh = Hist::bin(int16_t(mep + iphicut));
      auto incr = [](auto& k) { return k = (k + 1) % Hist::nbins(); };

#ifdef GPU_DEBUG
      int tot = 0;
      int nmin = 0;
      int tooMany = 0;
#endif

      auto khh = kh;
      incr(khh);
      for (auto kk = kl; kk != khh; incr(kk)) {
#ifdef GPU_DEBUG
        if (kk != kl && kk != kh)
          nmin += hist.size(kk + hoff);
#endif
        auto const* __restrict__ p = hist.begin(kk + hoff);
        auto const* __restrict__ e = hist.end(kk + hoff);
        p += first;
        for (; p < e; p += stride) {
          auto oi = __ldg(p);
          assert(oi >= offsets[outer]);
          assert(oi < offsets[outer + 1]);
          auto mo = hh.detectorIndex(oi);
          if (mo > 2000)
            continue;  //    invalid
          auto mop = hh.iphi(oi);
          if (std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop))) > iphicut)
            continue;
          if (doPhiCut) {
            if (doClusterCut && zsizeCut(oi))
              continue;
            if (z0cutoff(oi) || ptcut(oi, mop))
              continue;
          }
          auto ind = atomicAdd(nCells, 1);
          if (ind >= maxNumOfDoublets) {
            atomicSub(nCells, 1);
            break;
          }  // move to SimpleVector??
          // int layerPairId, int doubletId, int innerHitId, int outerHitId)
          cells[ind].init(/* *cellNeighbors, *cellTracks, */ hh, pairLayerId, ind, i, oi);
          isOuterHitOfCell[oi].push_back(ind);
#ifdef GPU_DEBUG
          if (isOuterHitOfCell[oi].full())
            ++tooMany;
          ++tot;
#endif
        }
      }
#ifdef GPU_DEBUG
      if (tooMany > 0)
        printf("OuterHitOfCell full for %d in layer %d/%d, %d,%d %d\n", i, inner, outer, nmin, tot, tooMany);
#endif
    }  // loop in block...
  }

}  // namespace gpuPixelDoubletsAlgos

namespace gpuPixelDoublets {
  using namespace gpuPixelDoubletsAlgos;

  constexpr int nPairs = 13 + 2 + 4;

  // start constants
  // clang-format off

  __constant__ const uint8_t layerPairs[2 * nPairs] = {
      0, 1, 0, 4, 0, 7,              // BPIX1 (3)
      1, 2, 1, 4, 1, 7,              // BPIX2 (5)
      4, 5, 7, 8,                    // FPIX1 (8)
      2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  // BPIX3 & FPIX2 (13)
      0, 2, 1, 3,                    // Jumping Barrel (15)
      0, 5, 0, 8,                    // Jumping Forward (BPIX1,FPIX2)
      4, 6, 7, 9                     // Jumping Forward (19)
  };

  constexpr int16_t phi0p05 = 522;  // round(521.52189...) = phi2short(0.05);
  constexpr int16_t phi0p06 = 626;  // round(625.82270...) = phi2short(0.06);
  constexpr int16_t phi0p07 = 730;  // round(730.12648...) = phi2short(0.07);

  __constant__ const int16_t phicuts[nPairs]{phi0p05,
                                             phi0p07,
                                             phi0p07,
                                             phi0p05,
                                             phi0p06,
                                             phi0p06,
                                             phi0p05,
                                             phi0p05,
                                             phi0p06,
                                             phi0p06,
                                             phi0p06,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05};
  //   phi0p07, phi0p07, phi0p06,phi0p06, phi0p06,phi0p06};  // relaxed cuts

  __constant__ float const minz[nPairs] = {
      -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.};
  __constant__ float const maxz[nPairs] = {
      20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};
  __constant__ float const maxr[nPairs] = {
      20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};

  // end constants
  // clang-format on

  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  __global__ void initDoublets(GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                               int nHits,
                               CellNeighborsVector* cellNeighbors,
                               CellNeighbors* cellNeighborsContainer,
                               CellTracksVector* cellTracks,
                               CellTracks* cellTracksContainer) {
    assert(isOuterHitOfCell);
    int first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = first; i < nHits; i += gridDim.x * blockDim.x)
      isOuterHitOfCell[i].reset();
  }

  constexpr auto getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
  constexpr auto getDoubletsFromHistoMinBlocksPerMP = 16;

  __global__
  __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)
      void getDoubletsFromHisto(GPUCACell* cells,
                                uint32_t* nCells,
                                //CellNeighborsVector* cellNeighbors,
                                //CellTracksVector* cellTracks,
                                TrackingRecHit2DSOAView const* __restrict__ hhp,
                                GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                int nActualPairs,
                                bool ideal_cond,
                                bool doClusterCut,
                                bool doZCut,
                                bool doPhiCut,
                                uint32_t maxNumOfDoublets) {
    auto const& __restrict__ hh = *hhp;
    doubletsFromHisto(layerPairs,
                      nActualPairs,
                      cells,
                      nCells,
                      //cellNeighbors,
                      //cellTracks,
                      hh,
                      isOuterHitOfCell,
                      phicuts,
                      minz,
                      maxz,
                      maxr,
                      ideal_cond,
                      doClusterCut,
                      doZCut,
                      doPhiCut,
                      maxNumOfDoublets);
  }
}  // namespace gpuPixelDoublets

int main() {

  cudaStream_t stream;
  cudaCheck(cudaStreamCreate(&stream));

  const unsigned int nActualPairs = 13;
  const bool idealConditions = true;
  const bool doClusterCut = true;
  const bool doZCut = true;
  const bool doPhiCut = true;
  const unsigned int maxNumberOfDoublets = 458752;
  const unsigned int n16 = 4;
  const unsigned int n32 = 9;

  uint32_t* nHits_ptr;
  uint32_t nHits;
  GPUCACell* theCells;
  GPUCACell* theCells_result;
  uint32_t* nCells;
  uint32_t* nCells_result;
  uint16_t* hh_16;
  float* hh_32;
  TrackingRecHit2DSOAView::Hist* hist;
  TrackingRecHit2DSOAView::AverageGeometry* geom;
  GPUCACell::OuterHitOfCell* isOuterHitOfCell;
  GPUCACell::OuterHitOfCell* isOuterHitOfCell_result;
  cudaCheck(cudaMallocHost(&nHits_ptr, sizeof(uint32_t)));
  cudaCheck(cudaMallocHost(&theCells, sizeof(GPUCACell)*maxNumberOfDoublets));
  cudaCheck(cudaMallocHost(&theCells_result, sizeof(GPUCACell)*maxNumberOfDoublets));
  cudaCheck(cudaMallocHost(&nCells, sizeof(uint32_t)));
  cudaCheck(cudaMallocHost(&nCells_result, sizeof(uint32_t)));
  cudaCheck(cudaMallocHost(&hist, sizeof(TrackingRecHit2DSOAView::Hist)));
  cudaCheck(cudaMallocHost(&geom, sizeof(TrackingRecHit2DSOAView::AverageGeometry)));
  {
    std::ifstream f;
    f.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
    f.open("dump.bin", std::ios_base::binary);
    f.read(reinterpret_cast<char *>(nHits_ptr), sizeof(uint32_t));
    nHits = *nHits_ptr;
    
    cudaCheck(cudaMallocHost(&hh_16, sizeof(uint16_t)*n16*nHits));
    cudaCheck(cudaMallocHost(&hh_32, sizeof(float)*(n32*nHits+11)));
    cudaCheck(cudaMallocHost(&isOuterHitOfCell, sizeof(GPUCACell::OuterHitOfCell)*std::max(1U, nHits)));
    cudaCheck(cudaMallocHost(&isOuterHitOfCell_result, sizeof(GPUCACell::OuterHitOfCell)*std::max(1U, nHits)));

    f.read(reinterpret_cast<char *>(theCells), sizeof(GPUCACell)*maxNumberOfDoublets);
    f.read(reinterpret_cast<char *>(nCells), sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(hh_16), sizeof(uint16_t)*n16*nHits);
    f.read(reinterpret_cast<char *>(hh_32), sizeof(float)*(n32*nHits+11));
    f.read(reinterpret_cast<char *>(hist), sizeof(TrackingRecHit2DSOAView::Hist));
    f.read(reinterpret_cast<char *>(geom), sizeof(TrackingRecHit2DSOAView::AverageGeometry));
    f.read(reinterpret_cast<char *>(isOuterHitOfCell), sizeof(GPUCACell::OuterHitOfCell)*std::max(1U, nHits));

    f.read(reinterpret_cast<char *>(theCells_result), sizeof(GPUCACell)*maxNumberOfDoublets);
    f.read(reinterpret_cast<char *>(nCells_result), sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(isOuterHitOfCell_result), sizeof(GPUCACell::OuterHitOfCell)*std::max(1U, nHits));
  }

  uint32_t* device_nHits_ptr;
  GPUCACell* device_theCells;
  uint32_t* device_nCells;
  uint16_t* device_hh_16;
  float* device_hh_32;
  TrackingRecHit2DSOAView::Hist* device_hist;
  TrackingRecHit2DSOAView::AverageGeometry* device_geom;
  GPUCACell::OuterHitOfCell* device_isOuterHitOfCell;

  cudaCheck(cudaMalloc(&device_nHits_ptr, sizeof(uint32_t)));
  cudaCheck(cudaMalloc(&device_theCells, sizeof(GPUCACell)*maxNumberOfDoublets));
  cudaCheck(cudaMalloc(&device_nCells, sizeof(uint32_t)));
  cudaCheck(cudaMalloc(&device_hist, sizeof(TrackingRecHit2DSOAView::Hist)));
  cudaCheck(cudaMalloc(&device_geom, sizeof(TrackingRecHit2DSOAView::AverageGeometry)));
  cudaCheck(cudaMalloc(&device_hh_16, sizeof(uint16_t)*n16*nHits));
  cudaCheck(cudaMalloc(&device_hh_32, sizeof(float)*(n32*nHits+11)));
  cudaCheck(cudaMalloc(&device_isOuterHitOfCell, sizeof(GPUCACell::OuterHitOfCell)*std::max(1U, nHits)));

  cudaCheck(cudaMemcpyAsync(device_nHits_ptr, nHits_ptr, sizeof(uint32_t), cudaMemcpyDefault, stream));
  cudaCheck(cudaMemcpyAsync(device_theCells, theCells, sizeof(GPUCACell)*maxNumberOfDoublets, cudaMemcpyDefault, stream));
  cudaCheck(cudaMemcpyAsync(device_nCells, nCells, sizeof(uint32_t), cudaMemcpyDefault, stream));
  cudaCheck(cudaMemcpyAsync(device_hist, hist, sizeof(TrackingRecHit2DSOAView::Hist), cudaMemcpyDefault, stream));
  cudaCheck(cudaMemcpyAsync(device_geom, geom, sizeof(TrackingRecHit2DSOAView::AverageGeometry), cudaMemcpyDefault, stream));
  cudaCheck(cudaMemcpyAsync(device_hh_16, hh_16, sizeof(uint16_t)*n16*nHits, cudaMemcpyDefault, stream));
  cudaCheck(cudaMemcpyAsync(device_hh_32, hh_32, sizeof(float)*(n32*nHits+11), cudaMemcpyDefault, stream));
  cudaCheck(cudaMemcpyAsync(device_isOuterHitOfCell, isOuterHitOfCell, sizeof(GPUCACell::OuterHitOfCell)*std::max(1U, nHits), cudaMemcpyDefault, stream));


  TrackingRecHit2DSOAView* view;
  TrackingRecHit2DSOAView* device_view;
  cudaCheck(cudaMallocHost(&view, sizeof(TrackingRecHit2DSOAView)));
  cudaCheck(cudaMalloc(&device_view, sizeof(TrackingRecHit2DSOAView)));

  view->m_nHits = nHits;
  view->m_averageGeometry = device_geom;

  auto get16 = [&](int i) { return device_hh_16 + i * nHits; };
  auto get32 = [&](int i) { return device_hh_32 + i * nHits; };

  view->m_hist = device_hist;

  view->m_xl = get32(0);
  view->m_yl = get32(1);
  view->m_xerr = get32(2);
  view->m_yerr = get32(3);

  view->m_xg = get32(4);
  view->m_yg = get32(5);
  view->m_zg = get32(6);
  view->m_rg = get32(7);

  view->m_iphi = reinterpret_cast<int16_t*>(get16(0));

  view->m_charge = reinterpret_cast<int32_t*>(get32(8));
  view->m_xsize = reinterpret_cast<int16_t*>(get16(2));
  view->m_ysize = reinterpret_cast<int16_t*>(get16(3));
  view->m_detInd = get16(1);
  view->m_hitsLayerStart = reinterpret_cast<uint32_t*>(get32(n32));

  cudaCheck(cudaMemcpyAsync(device_view, view, sizeof(TrackingRecHit2DSOAView), cudaMemcpyDefault, stream));
  cudaCheck(cudaStreamSynchronize(stream));

  std::cout << "Launching kernel" << std::endl;

  int stride = 1;
  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
  int blocks = (2 * nHits + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blks(1, blocks, 1);
  dim3 thrs(stride, threadsPerBlock, 1);
  gpuPixelDoublets::getDoubletsFromHisto<<<blks, thrs, 0, stream>>>(device_theCells,
                                                                    device_nCells,
                                                                    //device_theCellNeighbors_,
                                                                    //device_theCellTracks_,
                                                                    device_view,
                                                                    device_isOuterHitOfCell,
                                                                    nActualPairs,
                                                                    idealConditions,
                                                                    doClusterCut,
                                                                    doZCut,
                                                                    doPhiCut,
                                                                    maxNumberOfDoublets);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaMemcpyAsync(theCells, device_theCells, sizeof(GPUCACell)*maxNumberOfDoublets, cudaMemcpyDefault, stream));
  cudaCheck(cudaMemcpyAsync(nCells, device_nCells, sizeof(uint32_t), cudaMemcpyDefault, stream));
  cudaCheck(cudaMemcpyAsync(isOuterHitOfCell, device_isOuterHitOfCell, sizeof(GPUCACell::OuterHitOfCell)*std::max(1U, nHits), cudaMemcpyDefault, stream));

  cudaCheck(cudaStreamSynchronize(stream));
  std::cout << "Done, validating" << std::endl;
  bool validated = true;

  if(*nCells != *nCells_result) {
    std::cout << "nCells " << *nCells << " != " << *nCells_result << std::endl;
    validated = false;
  }
  /*
  for(size_t i=0; i<maxNumberOfDoublets; ++i) {
    if(theCells[i].get_inner_hit_id() != theCells_result[i].get_inner_hit_id()) {
      std::cout << "theCells[" << i << "].get_inner_hit_id() " << theCells[i].get_inner_hit_id() << " != " << theCells_result[i].get_inner_hit_id() << std::endl;
      validated = false;
      break;
    }
    if(theCells[i].get_outer_hit_id() != theCells_result[i].get_outer_hit_id()) {
      std::cout << "theCells[" << i << "].get_outer_hit_id() " << theCells[i].get_outer_hit_id() << " != " << theCells_result[i].get_outer_hit_id() << std::endl;
      validated = false;
    }
  }
  */
  for(size_t i=0; i<std::max(1U, nHits); ++i) {
    if(isOuterHitOfCell[i].size() != isOuterHitOfCell_result[i].size()) {
      std::cout << "isOuterHitOfCell[" << i << "].size() " << isOuterHitOfCell[i].size() << " != " << isOuterHitOfCell_result[i].size() << std::endl;
      validated = false;
    }
    /*
    else {
      for(size_t j=0; j<isOuterHitOfCell[i].size(); ++j) {
        if(isOuterHitOfCell[i][j] != isOuterHitOfCell_result[i][j]) {
          std::cout << "isOuterHitOfCell[" << i << "][" << j << "] " << isOuterHitOfCell[i][j] << " != " << isOuterHitOfCell_result[i][j] << std::endl;
        validated = false;
        }
      }
    }
    */
  }

  if(validated) {
    std::cout << "Results are OK!" << std::endl;
  }
  else {
    std::cout << "Validation FAILED!" << std::endl;
  }

  cudaCheck(cudaFree(device_view));
  cudaCheck(cudaFree(device_isOuterHitOfCell));
  cudaCheck(cudaFree(device_hh_16));
  cudaCheck(cudaFree(device_hh_32));
  cudaCheck(cudaFree(device_geom));
  cudaCheck(cudaFree(device_hist));
  cudaCheck(cudaFree(device_theCells));
  cudaCheck(cudaFree(device_nHits_ptr));

  cudaCheck(cudaFreeHost(view));
  cudaCheck(cudaFreeHost(isOuterHitOfCell_result));
  cudaCheck(cudaFreeHost(isOuterHitOfCell));
  cudaCheck(cudaFreeHost(hh_16));
  cudaCheck(cudaFreeHost(hh_32));
  cudaCheck(cudaFreeHost(geom));
  cudaCheck(cudaFreeHost(hist));
  cudaCheck(cudaFreeHost(nCells_result));
  cudaCheck(cudaFreeHost(nCells));
  cudaCheck(cudaFreeHost(theCells_result));
  cudaCheck(cudaFreeHost(theCells));
  cudaCheck(cudaFreeHost(nHits_ptr));

  return 0;
}
