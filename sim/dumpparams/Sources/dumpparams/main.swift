// Dump each policy's canonicalParameterBytes, using duckkit's own parser.
//
// WHY SWIFT AND NOT A JS ONNX READER. The byte layout the browser bench reads
// is DEFINED by DuckPolicy.canonicalParameterBytes — normalizer mean, then
// std, then per layer weights-then-biases, little-endian float32. A second
// implementation of that layout in JavaScript would be a second opinion about
// the contract, and the first time the two disagreed the phone would run a
// different network from the one the app fingerprinted. This is the contract,
// executed.
import Foundation
import DuckKit

let args = Array(CommandLine.arguments.dropFirst())
guard args.count == 2 else {
    FileHandle.standardError.write("usage: dumpparams <onnx-dir> <out-dir>\n".data(using: .utf8)!)
    exit(2)
}
let inDir = URL(fileURLWithPath: args[0]), outDir = URL(fileURLWithPath: args[1])
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

var names: [String] = []
for entry in try FileManager.default.contentsOfDirectory(atPath: inDir.path).sorted()
    where entry.hasSuffix(".onnx") && !entry.hasPrefix("uploaded-") {
    names.append(entry)
}
let community = inDir.appendingPathComponent("community")
if let dirs = try? FileManager.default.contentsOfDirectory(atPath: community.path).sorted() {
    for d in dirs where FileManager.default.fileExists(atPath: community.appendingPathComponent("\(d)/policy.onnx").path) {
        names.append("community/\(d)/policy.onnx")
    }
}

for name in names {
    let url = inDir.appendingPathComponent(name)
    do {
        let policy = try DuckPolicy.load(contentsOf: url)
        let bytes = policy.canonicalParameterBytes
        // The name the bench asks for: a community policy is `<dir>/policy.onnx`
        // to /policy, and a file whose name has a slash in it is not a file, so
        // the slash becomes a dash on disk and the manifest carries both.
        let flat = name.replacingOccurrences(of: "community/", with: "")
                       .replacingOccurrences(of: "/", with: "-")
        let out = outDir.appendingPathComponent(flat.replacingOccurrences(of: ".onnx", with: ".bin"))
        try bytes.write(to: out)
        let widths = policy.layerWidths.map { "\($0.inputs)x\($0.outputs)" }.joined(separator: " ")
        print("\(name)\t\(bytes.count)\t\(widths)\t\(policy.parameterCount)")
    } catch {
        print("\(name)\tREFUSED\t\(error)")
    }
}
