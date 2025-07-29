import CoreML

let argv = ProcessInfo.processInfo.arguments

guard argv.count == 2 else {
    print("Usage: swift compile_coreml_model.swift <path_to_model.mlpackage>")
    exit(1)
}

let modelURL = URL(fileURLWithPath: argv[1], isDirectory: true)
let outputURL = modelURL.deletingPathExtension().appendingPathExtension("mlmodelc")

let fileManager = FileManager.default

let compiledModelURL = try MLModel.compileModel(at: modelURL)

_ = try fileManager.replaceItemAt(outputURL, withItemAt: compiledModelURL)
