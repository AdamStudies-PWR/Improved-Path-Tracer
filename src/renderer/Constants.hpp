#pragma once

#include <cstdint>


namespace tracer::renderer
{

namespace
{
// 1024 is MAX
const uint32_t THREAD_SIZE = 256;
const uint32_t BLOCK_SIZE = 1024;
}  // namesapce

}  // namespace tracer::renderer
