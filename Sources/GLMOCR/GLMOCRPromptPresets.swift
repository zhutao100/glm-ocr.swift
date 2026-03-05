public enum GLMOCRPromptPresets {

    public static let `default` = """
        Recognize the text in the image and output in Markdown format.
        Preserve the original layout (headings/paragraphs/tables/formulas).
        Do not fabricate content that does not exist in the image.
        """

    public static let textRecognition = "Text Recognition:"
    public static let tableRecognition = "Table Recognition:"
    public static let formulaRecognition = "Formula Recognition:"
}
