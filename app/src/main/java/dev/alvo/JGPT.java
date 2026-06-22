package dev.alvo;

import dev.alvo.cli.JGPTCommand;

import picocli.CommandLine;

public class JGPT {

  static void main(String[] args) {
    trapExceptions();

    int exitCode = new CommandLine(new JGPTCommand())
      // Allow lowercase --type values (e.g. "safetensors") to match the uppercase enum constants.
      .setCaseInsensitiveEnumValuesAllowed(true)
      .execute(args);

    System.exit(exitCode);
  }

  private static void trapExceptions() {
    Thread.setDefaultUncaughtExceptionHandler((thread, throwable) ->
      System.err.println("Uncaught exception in thread " + thread.getName() + ": " + throwable.getMessage()));
  }
}
