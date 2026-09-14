/**
 * Small platform layer used by the Windows build of ORB-SLAM3.
 * The upstream project uses POSIX usleep(); MSVC provides Sleep() instead.
 */
#ifndef ORB_SLAM3_PLATFORM_H
#define ORB_SLAM3_PLATFORM_H

#ifdef _WIN32
#include <windows.h>
inline void orbslam3_usleep(unsigned int microseconds)
{
    ::Sleep((microseconds + 999U) / 1000U);
}
#else
#include <unistd.h>
inline void orbslam3_usleep(unsigned int microseconds)
{
    ::usleep(microseconds);
}
#endif

#ifndef usleep
#define usleep orbslam3_usleep
#endif

#endif // ORB_SLAM3_PLATFORM_H
