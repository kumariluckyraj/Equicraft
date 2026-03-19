/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "images.unsplash.com",
        pathname: "/**",
      },
    ],
  },

  // Prevent bundling @xenova/transformers (and other native binaries)
  webpack: (config, { isServer }) => {
    if (isServer) {
      // If config.externals is a function, handle both function and array
      if (typeof config.externals === "function") {
        const originalExternals = config.externals;
        config.externals = async (context, request, callback) => {
          if (request === "@xenova/transformers") return callback(null, "commonjs " + request);
          return originalExternals(context, request, callback);
        };
      } else {
        config.externals = config.externals || [];
        config.externals.push("@xenova/transformers");
      }
    }
    return config;
  },
};

module.exports = nextConfig;