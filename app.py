from astroframe.llm import ask_model


def main():
    print("Testing AstroFrame local AI connection...")

    response = ask_model(
        "Explain FWHM in astrophotography in one sentence."
    )

    print()
    print("GPT-OSS response:")
    print(response)


if __name__ == "__main__":
    main()
