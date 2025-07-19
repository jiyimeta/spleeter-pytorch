import CoreML

let fileManager = FileManager.default

let outputDirectoryURL = URL(string: "output/coreml/")!
let modelURL = outputDirectoryURL.appendingPathComponent("SpleeterModel.mlpackage")
let compiledModelURL = try MLModel.compileModel(at: modelURL)
let outputURL = outputDirectoryURL.appendingPathComponent("SpleeterModel.mlmodelc")

try fileManager.replaceItemAt(outputURL, withItemAt: compiledModelURL)
