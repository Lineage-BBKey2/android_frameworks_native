/*
 * Temporary GLES layer for the BlackBerry SDM660 EGLImage investigation.
 *
 * This records texture-name reuse, storage mutations, context share groups,
 * and live EGLImages. It logs snapshots for successful and failed
 * EGL_GL_TEXTURE_2D_KHR image creation so otherwise-identical allocations
 * can be compared around the first failure.
 */

#define LOG_TAG "BBRYGlesLifecycle"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <inttypes.h>
#include <log/log.h>
#include <string.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_map>

using EGLFuncPointer = __eglMustCastToProperFunctionPointerType;
using PFNEGLGETNEXTLAYERPROCADDRESSPROC = void* (*)(void*, const char*);

namespace {

constexpr size_t kMaxTextureRecords = 4096;
constexpr size_t kMaxImageRecords = 4096;

using PFN_glGenTextures = void(GL_APIENTRYP)(GLsizei, GLuint*);
using PFN_glDeleteTextures = void(GL_APIENTRYP)(GLsizei, const GLuint*);
using PFN_glGetIntegerv = void(GL_APIENTRYP)(GLenum, GLint*);
using PFN_glTexImage2D = void(GL_APIENTRYP)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum,
                                            GLenum, const void*);
using PFN_glTexSubImage2D = void(GL_APIENTRYP)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei,
                                               GLenum, GLenum, const void*);
using PFN_glCompressedTexImage2D = void(GL_APIENTRYP)(GLenum, GLint, GLenum, GLsizei, GLsizei,
                                                      GLint, GLsizei, const void*);
using PFN_glCopyTexImage2D = void(GL_APIENTRYP)(GLenum, GLint, GLenum, GLint, GLint, GLsizei,
                                                GLsizei, GLint);
using PFN_glCopyTexSubImage2D = void(GL_APIENTRYP)(GLenum, GLint, GLint, GLint, GLint, GLint,
                                                   GLsizei, GLsizei);
using PFN_glTexStorage2D = void(GL_APIENTRYP)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
using PFN_glTexStorage2DEXT = void(GL_APIENTRYP)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
using PFN_glTextureStorage2DEXT = void(GL_APIENTRYP)(GLuint, GLenum, GLsizei, GLenum, GLsizei,
                                                     GLsizei);
using PFN_glEGLImageTargetTexture2DOES = void(GL_APIENTRYP)(GLenum, GLeglImageOES);
using PFN_glEGLImageTargetTexStorageEXT = void(GL_APIENTRYP)(GLenum, GLeglImageOES, const GLint*);
using PFN_glEGLImageTargetTextureStorageEXT = void(GL_APIENTRYP)(GLuint, GLeglImageOES,
                                                                 const GLint*);

using PFN_eglCreateImage = EGLImage(EGLAPIENTRYP)(EGLDisplay, EGLContext, EGLenum, EGLClientBuffer,
                                                  const EGLAttrib*);
using PFN_eglDestroyImage = EGLBoolean(EGLAPIENTRYP)(EGLDisplay, EGLImage);

PFNEGLCREATECONTEXTPROC gNextEglCreateContext = nullptr;
PFNEGLDESTROYCONTEXTPROC gNextEglDestroyContext = nullptr;
PFNEGLMAKECURRENTPROC gNextEglMakeCurrent = nullptr;
PFNEGLRELEASETHREADPROC gNextEglReleaseThread = nullptr;
PFNEGLCREATEIMAGEKHRPROC gNextEglCreateImageKHR = nullptr;
PFNEGLDESTROYIMAGEKHRPROC gNextEglDestroyImageKHR = nullptr;
PFN_eglCreateImage gNextEglCreateImage = nullptr;
PFN_eglDestroyImage gNextEglDestroyImage = nullptr;

PFN_glGenTextures gNextGlGenTextures = nullptr;
PFN_glDeleteTextures gNextGlDeleteTextures = nullptr;
PFN_glGetIntegerv gNextGlGetIntegerv = nullptr;
PFN_glTexImage2D gNextGlTexImage2D = nullptr;
PFN_glTexSubImage2D gNextGlTexSubImage2D = nullptr;
PFN_glCompressedTexImage2D gNextGlCompressedTexImage2D = nullptr;
PFN_glCopyTexImage2D gNextGlCopyTexImage2D = nullptr;
PFN_glCopyTexSubImage2D gNextGlCopyTexSubImage2D = nullptr;
PFN_glTexStorage2D gNextGlTexStorage2D = nullptr;
PFN_glTexStorage2DEXT gNextGlTexStorage2DEXT = nullptr;
PFN_glTextureStorage2DEXT gNextGlTextureStorage2DEXT = nullptr;
PFN_glEGLImageTargetTexture2DOES gNextGlEGLImageTargetTexture2DOES = nullptr;
PFN_glEGLImageTargetTexStorageEXT gNextGlEGLImageTargetTexStorageEXT = nullptr;
PFN_glEGLImageTargetTextureStorageEXT gNextGlEGLImageTargetTextureStorageEXT = nullptr;

struct ContextRecord {
    uint64_t group = 0;
    EGLContext shareContext = EGL_NO_CONTEXT;
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLConfig config = nullptr;
    uint64_t createNs = 0;
    uint64_t lastCurrentNs = 0;
    uint64_t makeCurrentCount = 0;
    uint64_t unbindCount = 0;
    uint64_t migrationCount = 0;
    pid_t createTid = 0;
    pid_t firstCurrentTid = 0;
    pid_t lastCurrentTid = 0;
    EGLSurface lastDraw = EGL_NO_SURFACE;
    EGLSurface lastRead = EGL_NO_SURFACE;
    EGLint clientVersion = 1;
    EGLint minorVersion = 0;
    EGLint flags = 0;
    EGLint noError = 0;
    EGLint resetStrategy = 0;
    EGLint priority = 0;
    uint32_t attributeCount = 0;
    uint64_t attributeHash = 14695981039346656037ULL;
    bool createObserved = false;
    bool destroyRequested = false;
};

struct TextureKey {
    uint64_t group;
    GLuint name;

    bool operator==(const TextureKey& other) const {
        return group == other.group && name == other.name;
    }
};

struct TextureKeyHash {
    size_t operator()(const TextureKey& key) const {
        const size_t h1 = std::hash<uint64_t>{}(key.group);
        const size_t h2 = std::hash<GLuint>{}(key.name);
        return h1 ^ (h2 + 0x9e3779b9U + (h1 << 6) + (h1 >> 2));
    }
};

struct TextureRecord {
    uint32_t generation = 0;
    uint32_t reuseCount = 0;
    uint32_t generateCalls = 0;
    uint32_t deleteCalls = 0;
    uint32_t storageCalls = 0;
    uint32_t uploadCalls = 0;
    uint32_t imageSuccesses = 0;
    uint32_t imageFailures = 0;
    uint32_t imageDestroys = 0;
    uint32_t liveImages = 0;
    uint32_t retiredLiveImages = 0;

    bool live = false;
    bool generated = false;
    bool deleteRequested = false;
    bool pixelsPresent = false;

    const char* lastOp = "none";
    uint64_t lastSequence = 0;
    uint64_t lastMutationNs = 0;
    uint64_t lastImageSuccessNs = 0;
    pid_t lastTid = 0;
    EGLContext lastContext = EGL_NO_CONTEXT;

    GLenum target = 0;
    GLint level = -1;
    GLsizei levels = 0;
    GLenum internalFormat = 0;
    GLsizei width = -1;
    GLsizei height = -1;
    GLenum format = 0;
    GLenum type = 0;
};

struct ImageRecord {
    TextureKey texture{};
    uint32_t generation = 0;
};

std::mutex gLock;
std::unordered_map<uintptr_t, ContextRecord> gContexts;
std::unordered_map<TextureKey, TextureRecord, TextureKeyHash> gTextures;
std::unordered_map<uintptr_t, ImageRecord> gImages;
uint64_t gNextGroup = 1;
uint64_t gSequence = 0;
uint64_t gImageSuccessTotal = 0;
uint64_t gImageFailureTotal = 0;
uint64_t gImageDestroyTotal = 0;
uint64_t gLiveImageTotal = 0;
uint64_t gPeakLiveImageTotal = 0;

thread_local EGLContext gCurrentContext = EGL_NO_CONTEXT;
thread_local uint64_t gCurrentGroup = 0;
thread_local EGLSurface gCurrentDraw = EGL_NO_SURFACE;
thread_local EGLSurface gCurrentRead = EGL_NO_SURFACE;

void getThreadName(char (&name)[16]) {
    memset(name, 0, sizeof(name));
    if (prctl(PR_GET_NAME, name, 0, 0, 0) != 0 || !name[0]) {
        memcpy(name, "unknown", sizeof("unknown"));
    }
}

void parseContextAttributes(const EGLint* attributes, ContextRecord* record) {
    if (!record) return;

    // Bound the diagnostic parser even though EGL attribute lists are required
    // to be EGL_NONE-terminated.
    for (size_t pair = 0; attributes && pair < 32 && attributes[0] != EGL_NONE;
         pair++, attributes += 2) {
        record->attributeCount++;
        record->attributeHash ^= static_cast<uint32_t>(attributes[0]);
        record->attributeHash *= 1099511628211ULL;
        record->attributeHash ^= static_cast<uint32_t>(attributes[1]);
        record->attributeHash *= 1099511628211ULL;
        switch (attributes[0]) {
            case 0x3098: // EGL_CONTEXT_CLIENT_VERSION / EGL_CONTEXT_MAJOR_VERSION_KHR
                record->clientVersion = attributes[1];
                break;
            case 0x30FB: // EGL_CONTEXT_MINOR_VERSION_KHR
                record->minorVersion = attributes[1];
                break;
            case 0x30FC: // EGL_CONTEXT_FLAGS_KHR
                record->flags = attributes[1];
                break;
            case 0x31B3: // EGL_CONTEXT_OPENGL_NO_ERROR_KHR
                record->noError = attributes[1];
                break;
            case 0x31BD: // EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_KHR
                record->resetStrategy = attributes[1];
                break;
            case 0x3100: // EGL_CONTEXT_PRIORITY_LEVEL_IMG
                record->priority = attributes[1];
                break;
            default:
                break;
        }
    }
}

uint64_t nowNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
            static_cast<uint64_t>(ts.tv_nsec);
}

uint64_t ageUs(uint64_t timestampNs, uint64_t currentNs) {
    return timestampNs && currentNs >= timestampNs ? (currentNs - timestampNs) / 1000ULL : 0;
}

uint64_t groupForContextLocked(EGLContext context) {
    if (context == EGL_NO_CONTEXT) return 0;

    const uintptr_t key = reinterpret_cast<uintptr_t>(context);
    auto found = gContexts.find(key);
    if (found != gContexts.end()) return found->second.group;

    const uint64_t group = gNextGroup++;
    ContextRecord record;
    record.group = group;
    gContexts.emplace(key, record);
    return group;
}

void evictTextureIfNecessaryLocked() {
    if (gTextures.size() < kMaxTextureRecords) return;

    auto oldest = gTextures.end();
    for (auto it = gTextures.begin(); it != gTextures.end(); ++it) {
        if (it->second.liveImages != 0) continue;
        if (oldest == gTextures.end() || it->second.lastSequence < oldest->second.lastSequence) {
            oldest = it;
        }
    }
    if (oldest == gTextures.end()) {
        oldest = std::min_element(gTextures.begin(), gTextures.end(), [](const auto& a,
                                                                         const auto& b) {
            return a.second.lastSequence < b.second.lastSequence;
        });
    }
    if (oldest != gTextures.end()) gTextures.erase(oldest);
}

TextureRecord& textureRecordLocked(const TextureKey& key) {
    auto found = gTextures.find(key);
    if (found != gTextures.end()) return found->second;

    evictTextureIfNecessaryLocked();
    return gTextures.emplace(key, TextureRecord{}).first->second;
}

void beginNewGenerationLocked(TextureRecord& record, bool generated) {
    if (record.generation != 0) {
        record.reuseCount++;
        record.retiredLiveImages += record.liveImages;
    }

    record.generation++;
    record.generateCalls++;
    record.live = true;
    record.generated = generated;
    record.deleteRequested = false;
    record.storageCalls = 0;
    record.uploadCalls = 0;
    record.imageSuccesses = 0;
    record.imageFailures = 0;
    record.imageDestroys = 0;
    record.liveImages = 0;
    record.pixelsPresent = false;
    record.lastOp = generated ? "GenTextures" : "implicit";
    record.level = -1;
    record.levels = 0;
    record.internalFormat = 0;
    record.width = -1;
    record.height = -1;
    record.format = 0;
    record.type = 0;
    record.lastImageSuccessNs = 0;
}

GLuint boundTexture2D() {
    if (!gNextGlGetIntegerv || gCurrentContext == EGL_NO_CONTEXT) return 0;
    GLint binding = 0;
    gNextGlGetIntegerv(GL_TEXTURE_BINDING_2D, &binding);
    return binding > 0 ? static_cast<GLuint>(binding) : 0;
}

void recordMutation(GLuint texture, const char* op, GLenum target, GLint level, GLsizei levels,
                    GLenum internalFormat, GLsizei width, GLsizei height, GLenum format,
                    GLenum type, bool pixelsPresent, bool isStorage, bool isUpload) {
    if (!texture || !gCurrentGroup) return;

    const uint64_t timestamp = nowNs();
    std::lock_guard<std::mutex> guard(gLock);
    TextureRecord& record = textureRecordLocked(TextureKey{gCurrentGroup, texture});
    if (!record.live || record.deleteRequested) beginNewGenerationLocked(record, false);

    record.lastOp = op;
    record.lastSequence = ++gSequence;
    record.lastMutationNs = timestamp;
    record.lastTid = gettid();
    record.lastContext = gCurrentContext;
    record.pixelsPresent = pixelsPresent;
    if (isStorage) {
        record.storageCalls++;
        record.target = target;
        record.level = level;
        record.levels = levels;
        record.internalFormat = internalFormat;
        record.width = width;
        record.height = height;
        record.format = format;
        record.type = type;
    }
    if (isUpload) record.uploadCalls++;

    if (isStorage && (width == 0 || height == 0)) {
        ALOGE("BBRY_GL_TEX_ZERO_STORAGE ctx=%p group=%" PRIu64
              " texture=0x%x generation=%u op=%s level=%d levels=%d size=%dx%d seq=%" PRIu64,
              gCurrentContext, gCurrentGroup, texture, record.generation, op, level, levels, width,
              height, record.lastSequence);
    }
}

template <typename AttrType>
void recordImageResult(const char* api, EGLContext context, EGLenum target, EGLClientBuffer buffer,
                       const AttrType* attributes, EGLImageKHR image) {
    if (target != EGL_GL_TEXTURE_2D_KHR) return;

    const GLuint texture = static_cast<GLuint>(reinterpret_cast<uintptr_t>(buffer));
    if (!texture) return;

    GLint level = 0;
    EGLBoolean preserved = EGL_FALSE;
    for (const AttrType* attr = attributes; attr && attr[0] != EGL_NONE; attr += 2) {
        if (attr[0] == EGL_GL_TEXTURE_LEVEL_KHR) level = static_cast<GLint>(attr[1]);
        if (attr[0] == EGL_IMAGE_PRESERVED_KHR) {
            preserved = static_cast<EGLBoolean>(attr[1]);
        }
    }

    const uint64_t timestamp = nowNs();
    TextureRecord snapshot;
    bool foundRecord = false;
    uint64_t group = 0;
    uint64_t successTotal = 0;
    uint64_t failureTotal = 0;
    uint64_t destroyTotal = 0;
    uint64_t liveTotal = 0;
    uint64_t peakLiveTotal = 0;
    size_t trackedImages = 0;
    ContextRecord contextSnapshot;

    {
        std::lock_guard<std::mutex> guard(gLock);
        group = groupForContextLocked(context);
        auto contextIt = gContexts.find(reinterpret_cast<uintptr_t>(context));
        if (contextIt != gContexts.end()) {
            contextSnapshot = contextIt->second;
        }
        const TextureKey key{group, texture};
        auto found = gTextures.find(key);
        foundRecord = found != gTextures.end();
        TextureRecord& record = textureRecordLocked(key);

        if (image != EGL_NO_IMAGE_KHR) {
            record.imageSuccesses++;
            record.liveImages++;
            record.lastImageSuccessNs = timestamp;
            if (gImages.size() >= kMaxImageRecords) gImages.erase(gImages.begin());
            gImages[reinterpret_cast<uintptr_t>(image)] = ImageRecord{key, record.generation};
            gImageSuccessTotal++;
            gLiveImageTotal++;
            gPeakLiveImageTotal = std::max(gPeakLiveImageTotal, gLiveImageTotal);
        } else {
            record.imageFailures++;
            gImageFailureTotal++;
        }
        snapshot = record;
        successTotal = gImageSuccessTotal;
        failureTotal = gImageFailureTotal;
        destroyTotal = gImageDestroyTotal;
        liveTotal = gLiveImageTotal;
        peakLiveTotal = gPeakLiveImageTotal;
        trackedImages = gImages.size();
    }

    char threadName[16];
    getThreadName(threadName);

    if (image != EGL_NO_IMAGE_KHR) {
        ALOGI("BBRY_GL_TEX_IMAGE_OK api=%s ctx=%p currentCtx=%p ctxGroup=%" PRIu64
              " currentGroup=%" PRIu64 " texture=0x%x found=%d generation=%u reuse=%u "
              "lastOp=%s mutationAgeUs=%" PRIu64 " target=0x%x level=%d levels=%d "
              "internal=0x%x size=%dx%d format=0x%x type=0x%x pixels=%d storageCalls=%u "
              "uploads=%u imageSuccess=%u liveImages=%u requestLevel=%d preserved=%d "
              "globalSuccess=%" PRIu64 " globalFail=%" PRIu64 " globalDestroy=%" PRIu64
              " globalLive=%" PRIu64 " globalPeak=%" PRIu64 " trackedImages=%zu "
              "tid=%d thread=%s ctxCreated=%d ctxAgeUs=%" PRIu64 " config=%p "
              "clientVersion=%d minorVersion=%d flags=0x%x noError=%d reset=0x%x "
              "priority=0x%x attrCount=%u attrHash=0x%" PRIx64
              " makeCurrent=%" PRIu64 " unbind=%" PRIu64
              " migrations=%" PRIu64 " firstCurrentTid=%d lastCurrentTid=%d "
              "currentAgeUs=%" PRIu64 " draw=%p read=%p currentDraw=%p currentRead=%p",
              api, context, gCurrentContext, group, gCurrentGroup, texture,
              foundRecord ? 1 : 0, snapshot.generation, snapshot.reuseCount, snapshot.lastOp,
              ageUs(snapshot.lastMutationNs, timestamp), snapshot.target, snapshot.level,
              snapshot.levels, snapshot.internalFormat, snapshot.width, snapshot.height,
              snapshot.format, snapshot.type, snapshot.pixelsPresent ? 1 : 0,
              snapshot.storageCalls, snapshot.uploadCalls, snapshot.imageSuccesses,
              snapshot.liveImages, level, preserved, successTotal, failureTotal, destroyTotal,
              liveTotal, peakLiveTotal, trackedImages, gettid(), threadName,
              contextSnapshot.createObserved ? 1 : 0,
              ageUs(contextSnapshot.createNs, timestamp),
              contextSnapshot.config, contextSnapshot.clientVersion,
              contextSnapshot.minorVersion, contextSnapshot.flags, contextSnapshot.noError,
              contextSnapshot.resetStrategy, contextSnapshot.priority,
              contextSnapshot.attributeCount, contextSnapshot.attributeHash,
              contextSnapshot.makeCurrentCount, contextSnapshot.unbindCount,
              contextSnapshot.migrationCount, contextSnapshot.firstCurrentTid,
              contextSnapshot.lastCurrentTid,
              ageUs(contextSnapshot.lastCurrentNs, timestamp), contextSnapshot.lastDraw,
              contextSnapshot.lastRead, gCurrentDraw, gCurrentRead);
        return;
    }

    ALOGE("BBRY_GL_TEX_IMAGE_FAIL api=%s ctx=%p currentCtx=%p ctxGroup=%" PRIu64
          " currentGroup=%" PRIu64 " texture=0x%x found=%d generation=%u reuse=%u live=%d "
          "generated=%d deleted=%d lastOp=%s lastCtx=%p lastTid=%d lastSeq=%" PRIu64
          " mutationAgeUs=%" PRIu64 " target=0x%x level=%d levels=%d internal=0x%x "
          "size=%dx%d format=0x%x type=0x%x pixels=%d storageCalls=%u uploads=%u "
          "imageSuccess=%u imageFail=%u liveImages=%u imageDestroy=%u retiredLiveImages=%u "
          "lastSuccessAgeUs=%" PRIu64 " requestLevel=%d preserved=%d globalSuccess=%" PRIu64
          " globalFail=%" PRIu64 " globalDestroy=%" PRIu64 " globalLive=%" PRIu64
          " globalPeak=%" PRIu64 " trackedImages=%zu tid=%d thread=%s ctxCreated=%d "
          "ctxAgeUs=%" PRIu64 " config=%p clientVersion=%d minorVersion=%d flags=0x%x "
          "noError=%d reset=0x%x priority=0x%x attrCount=%u attrHash=0x%" PRIx64
          " makeCurrent=%" PRIu64
          " unbind=%" PRIu64 " migrations=%" PRIu64 " firstCurrentTid=%d "
          "lastCurrentTid=%d currentAgeUs=%" PRIu64 " draw=%p read=%p currentDraw=%p "
          "currentRead=%p",
          api, context, gCurrentContext, group, gCurrentGroup, texture, foundRecord ? 1 : 0,
          snapshot.generation, snapshot.reuseCount, snapshot.live ? 1 : 0,
          snapshot.generated ? 1 : 0, snapshot.deleteRequested ? 1 : 0, snapshot.lastOp,
          snapshot.lastContext, snapshot.lastTid, snapshot.lastSequence,
          ageUs(snapshot.lastMutationNs, timestamp), snapshot.target, snapshot.level,
          snapshot.levels, snapshot.internalFormat, snapshot.width, snapshot.height,
          snapshot.format, snapshot.type, snapshot.pixelsPresent ? 1 : 0,
          snapshot.storageCalls, snapshot.uploadCalls, snapshot.imageSuccesses,
          snapshot.imageFailures, snapshot.liveImages, snapshot.imageDestroys,
          snapshot.retiredLiveImages, ageUs(snapshot.lastImageSuccessNs, timestamp), level,
          preserved, successTotal, failureTotal, destroyTotal, liveTotal, peakLiveTotal,
          trackedImages, gettid(), threadName, contextSnapshot.createObserved ? 1 : 0,
          ageUs(contextSnapshot.createNs, timestamp), contextSnapshot.config,
          contextSnapshot.clientVersion, contextSnapshot.minorVersion, contextSnapshot.flags,
          contextSnapshot.noError, contextSnapshot.resetStrategy, contextSnapshot.priority,
          contextSnapshot.attributeCount, contextSnapshot.attributeHash,
          contextSnapshot.makeCurrentCount, contextSnapshot.unbindCount,
          contextSnapshot.migrationCount, contextSnapshot.firstCurrentTid,
          contextSnapshot.lastCurrentTid, ageUs(contextSnapshot.lastCurrentNs, timestamp),
          contextSnapshot.lastDraw, contextSnapshot.lastRead, gCurrentDraw, gCurrentRead);
}

void recordImageDestroy(EGLImageKHR image, EGLBoolean result) {
    if (result != EGL_TRUE || image == EGL_NO_IMAGE_KHR) return;

    std::lock_guard<std::mutex> guard(gLock);
    auto imageIt = gImages.find(reinterpret_cast<uintptr_t>(image));
    if (imageIt == gImages.end()) return;

    auto textureIt = gTextures.find(imageIt->second.texture);
    if (textureIt != gTextures.end() &&
        textureIt->second.generation == imageIt->second.generation) {
        TextureRecord& record = textureIt->second;
        if (record.liveImages) record.liveImages--;
        record.imageDestroys++;
    }
    gImages.erase(imageIt);
    gImageDestroyTotal++;
    if (gLiveImageTotal) gLiveImageTotal--;
}

EGLContext EGLAPIENTRY layerEglCreateContext(EGLDisplay display, EGLConfig config,
                                             EGLContext shareContext,
                                             const EGLint* attributes) {
    EGLContext result = gNextEglCreateContext(display, config, shareContext, attributes);
    if (result == EGL_NO_CONTEXT) return result;

    const uint64_t timestamp = nowNs();
    ContextRecord record;
    record.shareContext = shareContext;
    record.display = display;
    record.config = config;
    record.createNs = timestamp;
    record.createTid = gettid();
    record.createObserved = true;
    parseContextAttributes(attributes, &record);

    uint64_t group;
    {
        std::lock_guard<std::mutex> guard(gLock);
        group = shareContext == EGL_NO_CONTEXT ? gNextGroup++ : groupForContextLocked(shareContext);
        record.group = group;
        gContexts[reinterpret_cast<uintptr_t>(result)] = record;
    }
    char threadName[16];
    getThreadName(threadName);
    ALOGI("BBRY_GL_CONTEXT_CREATE ctx=%p share=%p group=%" PRIu64
          " display=%p config=%p tid=%d thread=%s clientVersion=%d minorVersion=%d "
          "flags=0x%x noError=%d reset=0x%x priority=0x%x attrCount=%u attrHash=0x%" PRIx64,
          result, shareContext, group, display, config, gettid(), threadName,
          record.clientVersion, record.minorVersion, record.flags, record.noError,
          record.resetStrategy, record.priority, record.attributeCount, record.attributeHash);
    return result;
}

EGLBoolean EGLAPIENTRY layerEglDestroyContext(EGLDisplay display, EGLContext context) {
    EGLBoolean result = gNextEglDestroyContext(display, context);
    if (result == EGL_TRUE && context != EGL_NO_CONTEXT) {
        std::lock_guard<std::mutex> guard(gLock);
        auto found = gContexts.find(reinterpret_cast<uintptr_t>(context));
        if (found != gContexts.end()) found->second.destroyRequested = true;
    }
    return result;
}

EGLBoolean EGLAPIENTRY layerEglMakeCurrent(EGLDisplay display, EGLSurface draw, EGLSurface read,
                                           EGLContext context) {
    const EGLContext previousContext = gCurrentContext;
    EGLBoolean result = gNextEglMakeCurrent(display, draw, read, context);
    if (result == EGL_TRUE) {
        const uint64_t timestamp = nowNs();
        const pid_t tid = gettid();
        gCurrentContext = context;
        gCurrentDraw = draw;
        gCurrentRead = read;
        std::lock_guard<std::mutex> guard(gLock);
        if (previousContext != EGL_NO_CONTEXT && previousContext != context) {
            auto previous = gContexts.find(reinterpret_cast<uintptr_t>(previousContext));
            if (previous != gContexts.end()) previous->second.unbindCount++;
        }
        gCurrentGroup = groupForContextLocked(context);
        if (context != EGL_NO_CONTEXT) {
            ContextRecord& record = gContexts[reinterpret_cast<uintptr_t>(context)];
            record.makeCurrentCount++;
            if (!record.firstCurrentTid) record.firstCurrentTid = tid;
            if (record.lastCurrentTid && record.lastCurrentTid != tid) record.migrationCount++;
            record.lastCurrentTid = tid;
            record.lastCurrentNs = timestamp;
            record.lastDraw = draw;
            record.lastRead = read;
        }
    }
    return result;
}

EGLBoolean EGLAPIENTRY layerEglReleaseThread() {
    const EGLContext previousContext = gCurrentContext;
    EGLBoolean result = gNextEglReleaseThread();
    if (result == EGL_TRUE) {
        if (previousContext != EGL_NO_CONTEXT) {
            std::lock_guard<std::mutex> guard(gLock);
            auto previous = gContexts.find(reinterpret_cast<uintptr_t>(previousContext));
            if (previous != gContexts.end()) previous->second.unbindCount++;
        }
        gCurrentContext = EGL_NO_CONTEXT;
        gCurrentGroup = 0;
        gCurrentDraw = EGL_NO_SURFACE;
        gCurrentRead = EGL_NO_SURFACE;
    }
    return result;
}

EGLImageKHR EGLAPIENTRY layerEglCreateImageKHR(EGLDisplay display, EGLContext context,
                                               EGLenum target, EGLClientBuffer buffer,
                                               const EGLint* attributes) {
    EGLImageKHR result =
            gNextEglCreateImageKHR(display, context, target, buffer, attributes);
    recordImageResult("KHR", context, target, buffer, attributes, result);
    return result;
}

EGLBoolean EGLAPIENTRY layerEglDestroyImageKHR(EGLDisplay display, EGLImageKHR image) {
    EGLBoolean result = gNextEglDestroyImageKHR(display, image);
    recordImageDestroy(image, result);
    return result;
}

EGLImage EGLAPIENTRY layerEglCreateImage(EGLDisplay display, EGLContext context, EGLenum target,
                                         EGLClientBuffer buffer, const EGLAttrib* attributes) {
    EGLImage result = gNextEglCreateImage(display, context, target, buffer, attributes);
    recordImageResult("EGL15", context, target, buffer, attributes, result);
    return result;
}

EGLBoolean EGLAPIENTRY layerEglDestroyImage(EGLDisplay display, EGLImage image) {
    EGLBoolean result = gNextEglDestroyImage(display, image);
    recordImageDestroy(image, result);
    return result;
}

void GL_APIENTRY layerGlGenTextures(GLsizei count, GLuint* textures) {
    gNextGlGenTextures(count, textures);
    if (count <= 0 || !textures || !gCurrentGroup) return;

    const uint64_t timestamp = nowNs();
    std::lock_guard<std::mutex> guard(gLock);
    for (GLsizei i = 0; i < count; i++) {
        if (!textures[i]) continue;
        TextureRecord& record = textureRecordLocked(TextureKey{gCurrentGroup, textures[i]});
        beginNewGenerationLocked(record, true);
        record.lastSequence = ++gSequence;
        record.lastMutationNs = timestamp;
        record.lastTid = gettid();
        record.lastContext = gCurrentContext;
    }
}

void GL_APIENTRY layerGlDeleteTextures(GLsizei count, const GLuint* textures) {
    gNextGlDeleteTextures(count, textures);
    if (count <= 0 || !textures || !gCurrentGroup) return;

    const uint64_t timestamp = nowNs();
    std::lock_guard<std::mutex> guard(gLock);
    for (GLsizei i = 0; i < count; i++) {
        if (!textures[i]) continue;
        TextureRecord& record = textureRecordLocked(TextureKey{gCurrentGroup, textures[i]});
        record.deleteCalls++;
        record.deleteRequested = true;
        record.live = false;
        record.lastOp = "DeleteTextures";
        record.lastSequence = ++gSequence;
        record.lastMutationNs = timestamp;
        record.lastTid = gettid();
        record.lastContext = gCurrentContext;
        if (record.liveImages) {
            ALOGE("BBRY_GL_TEX_DELETE_LIVE ctx=%p group=%" PRIu64
                  " texture=0x%x generation=%u liveImages=%u seq=%" PRIu64,
                  gCurrentContext, gCurrentGroup, textures[i], record.generation,
                  record.liveImages, record.lastSequence);
        }
    }
}

void GL_APIENTRY layerGlTexImage2D(GLenum target, GLint level, GLint internalFormat,
                                   GLsizei width, GLsizei height, GLint border, GLenum format,
                                   GLenum type, const void* pixels) {
    const GLuint texture = target == GL_TEXTURE_2D ? boundTexture2D() : 0;
    gNextGlTexImage2D(target, level, internalFormat, width, height, border, format, type, pixels);
    recordMutation(texture, "TexImage2D", target, level, 0, internalFormat, width, height, format,
                   type, pixels != nullptr, true, false);
}

void GL_APIENTRY layerGlTexSubImage2D(GLenum target, GLint level, GLint xOffset, GLint yOffset,
                                      GLsizei width, GLsizei height, GLenum format, GLenum type,
                                      const void* pixels) {
    const GLuint texture = target == GL_TEXTURE_2D ? boundTexture2D() : 0;
    gNextGlTexSubImage2D(target, level, xOffset, yOffset, width, height, format, type, pixels);
    recordMutation(texture, "TexSubImage2D", target, level, 0, 0, width, height, format, type,
                   pixels != nullptr, false, true);
}

void GL_APIENTRY layerGlCompressedTexImage2D(GLenum target, GLint level, GLenum internalFormat,
                                             GLsizei width, GLsizei height, GLint border,
                                             GLsizei imageSize, const void* data) {
    const GLuint texture = target == GL_TEXTURE_2D ? boundTexture2D() : 0;
    gNextGlCompressedTexImage2D(target, level, internalFormat, width, height, border, imageSize,
                                data);
    recordMutation(texture, "CompressedTexImage2D", target, level, 0, internalFormat, width,
                   height, 0, 0, data != nullptr, true, false);
}

void GL_APIENTRY layerGlCopyTexImage2D(GLenum target, GLint level, GLenum internalFormat, GLint x,
                                       GLint y, GLsizei width, GLsizei height, GLint border) {
    const GLuint texture = target == GL_TEXTURE_2D ? boundTexture2D() : 0;
    gNextGlCopyTexImage2D(target, level, internalFormat, x, y, width, height, border);
    recordMutation(texture, "CopyTexImage2D", target, level, 0, internalFormat, width, height, 0,
                   0, false, true, false);
}

void GL_APIENTRY layerGlCopyTexSubImage2D(GLenum target, GLint level, GLint xOffset, GLint yOffset,
                                          GLint x, GLint y, GLsizei width, GLsizei height) {
    const GLuint texture = target == GL_TEXTURE_2D ? boundTexture2D() : 0;
    gNextGlCopyTexSubImage2D(target, level, xOffset, yOffset, x, y, width, height);
    recordMutation(texture, "CopyTexSubImage2D", target, level, 0, 0, width, height, 0, 0, false,
                   false, true);
}

void GL_APIENTRY layerGlTexStorage2D(GLenum target, GLsizei levels, GLenum internalFormat,
                                     GLsizei width, GLsizei height) {
    const GLuint texture = target == GL_TEXTURE_2D ? boundTexture2D() : 0;
    gNextGlTexStorage2D(target, levels, internalFormat, width, height);
    recordMutation(texture, "TexStorage2D", target, 0, levels, internalFormat, width, height, 0,
                   0, false, true, false);
}

void GL_APIENTRY layerGlTexStorage2DEXT(GLenum target, GLsizei levels, GLenum internalFormat,
                                        GLsizei width, GLsizei height) {
    const GLuint texture = target == GL_TEXTURE_2D ? boundTexture2D() : 0;
    gNextGlTexStorage2DEXT(target, levels, internalFormat, width, height);
    recordMutation(texture, "TexStorage2DEXT", target, 0, levels, internalFormat, width, height,
                   0, 0, false, true, false);
}

void GL_APIENTRY layerGlTextureStorage2DEXT(GLuint texture, GLenum target, GLsizei levels,
                                            GLenum internalFormat, GLsizei width, GLsizei height) {
    gNextGlTextureStorage2DEXT(texture, target, levels, internalFormat, width, height);
    recordMutation(texture, "TextureStorage2DEXT", target, 0, levels, internalFormat, width,
                   height, 0, 0, false, true, false);
}

void GL_APIENTRY layerGlEGLImageTargetTexture2DOES(GLenum target, GLeglImageOES image) {
    const GLuint texture = target == GL_TEXTURE_2D ? boundTexture2D() : 0;
    gNextGlEGLImageTargetTexture2DOES(target, image);
    recordMutation(texture, "EGLImageTargetTexture2DOES", target, 0, 0, 0, -1, -1, 0, 0,
                   image != nullptr, true, false);
}

void GL_APIENTRY layerGlEGLImageTargetTexStorageEXT(GLenum target, GLeglImageOES image,
                                                    const GLint* attributes) {
    const GLuint texture = target == GL_TEXTURE_2D ? boundTexture2D() : 0;
    gNextGlEGLImageTargetTexStorageEXT(target, image, attributes);
    recordMutation(texture, "EGLImageTargetTexStorageEXT", target, 0, 0, 0, -1, -1, 0, 0,
                   image != nullptr, true, false);
}

void GL_APIENTRY layerGlEGLImageTargetTextureStorageEXT(GLuint texture, GLeglImageOES image,
                                                        const GLint* attributes) {
    gNextGlEGLImageTargetTextureStorageEXT(texture, image, attributes);
    recordMutation(texture, "EGLImageTargetTextureStorageEXT", GL_TEXTURE_2D, 0, 0, 0, -1, -1,
                   0, 0, image != nullptr, true, false);
}

template <typename FunctionType>
EGLFuncPointer intercept(const char* requestedName, const char* expectedName, EGLFuncPointer next,
                         FunctionType* nextStorage, EGLFuncPointer wrapper) {
    if (strcmp(requestedName, expectedName) != 0) return nullptr;
    *nextStorage = reinterpret_cast<FunctionType>(next);
    return wrapper;
}

} // namespace

extern "C" {

__attribute__((visibility("default"))) EGLFuncPointer AndroidGLESLayer_Initialize(
        const void* layerId, PFNEGLGETNEXTLAYERPROCADDRESSPROC getNextLayerProcAddress) {
    if (layerId && getNextLayerProcAddress) {
        gNextGlGetIntegerv = reinterpret_cast<PFN_glGetIntegerv>(
                getNextLayerProcAddress(const_cast<void*>(layerId), "glGetIntegerv"));
    }
    ALOGI("BBRY_GL_LIFECYCLE_LAYER initialized maxTextures=%zu maxImages=%zu", kMaxTextureRecords,
          kMaxImageRecords);
    return nullptr;
}

__attribute__((visibility("default"))) EGLFuncPointer AndroidGLESLayer_GetProcAddress(
        const char* name, EGLFuncPointer next) {
#define BBRY_INTERCEPT(api, storage, wrapper)                                                \
    if (EGLFuncPointer result =                                                            \
                intercept(name, #api, next, &storage, reinterpret_cast<EGLFuncPointer>(wrapper))) { \
        return result;                                                                     \
    }

    BBRY_INTERCEPT(eglCreateContext, gNextEglCreateContext, layerEglCreateContext)
    BBRY_INTERCEPT(eglDestroyContext, gNextEglDestroyContext, layerEglDestroyContext)
    BBRY_INTERCEPT(eglMakeCurrent, gNextEglMakeCurrent, layerEglMakeCurrent)
    BBRY_INTERCEPT(eglReleaseThread, gNextEglReleaseThread, layerEglReleaseThread)
    BBRY_INTERCEPT(eglCreateImageKHR, gNextEglCreateImageKHR, layerEglCreateImageKHR)
    BBRY_INTERCEPT(eglDestroyImageKHR, gNextEglDestroyImageKHR, layerEglDestroyImageKHR)
    BBRY_INTERCEPT(eglCreateImage, gNextEglCreateImage, layerEglCreateImage)
    BBRY_INTERCEPT(eglDestroyImage, gNextEglDestroyImage, layerEglDestroyImage)

    BBRY_INTERCEPT(glGenTextures, gNextGlGenTextures, layerGlGenTextures)
    BBRY_INTERCEPT(glDeleteTextures, gNextGlDeleteTextures, layerGlDeleteTextures)
    BBRY_INTERCEPT(glTexImage2D, gNextGlTexImage2D, layerGlTexImage2D)
    BBRY_INTERCEPT(glTexSubImage2D, gNextGlTexSubImage2D, layerGlTexSubImage2D)
    BBRY_INTERCEPT(glCompressedTexImage2D, gNextGlCompressedTexImage2D,
                   layerGlCompressedTexImage2D)
    BBRY_INTERCEPT(glCopyTexImage2D, gNextGlCopyTexImage2D, layerGlCopyTexImage2D)
    BBRY_INTERCEPT(glCopyTexSubImage2D, gNextGlCopyTexSubImage2D, layerGlCopyTexSubImage2D)
    BBRY_INTERCEPT(glTexStorage2D, gNextGlTexStorage2D, layerGlTexStorage2D)
    BBRY_INTERCEPT(glTexStorage2DEXT, gNextGlTexStorage2DEXT, layerGlTexStorage2DEXT)
    BBRY_INTERCEPT(glTextureStorage2DEXT, gNextGlTextureStorage2DEXT,
                   layerGlTextureStorage2DEXT)
    BBRY_INTERCEPT(glEGLImageTargetTexture2DOES, gNextGlEGLImageTargetTexture2DOES,
                   layerGlEGLImageTargetTexture2DOES)
    BBRY_INTERCEPT(glEGLImageTargetTexStorageEXT, gNextGlEGLImageTargetTexStorageEXT,
                   layerGlEGLImageTargetTexStorageEXT)
    BBRY_INTERCEPT(glEGLImageTargetTextureStorageEXT, gNextGlEGLImageTargetTextureStorageEXT,
                   layerGlEGLImageTargetTextureStorageEXT)

#undef BBRY_INTERCEPT

    if (strcmp(name, "glGetIntegerv") == 0) {
        gNextGlGetIntegerv = reinterpret_cast<PFN_glGetIntegerv>(next);
    }
    return next;
}

} // extern "C"
