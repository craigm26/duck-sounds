// swift-tools-version:5.9
import PackageDescription
let package = Package(
    name: "dumpparams",
    platforms: [.macOS(.v13)],
    dependencies: [.package(path: "/home/craigm26/projects/duckkit")],
    targets: [.executableTarget(name: "dumpparams",
                                dependencies: [.product(name: "DuckKit", package: "duckkit")])]
)
