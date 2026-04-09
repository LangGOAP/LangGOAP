"""Shared execute functions for LangGoap tutorial notebooks and integration tests.

Each sub-module corresponds to one tutorial and contains the execute functions
and ActionSpec factory functions used by both the notebook (runnable documentation)
and the integration test (source of truth).

Import pattern
--------------
In tests::

    from tutorial_examples.adaptive_rag import adaptive_rag_actions, retrieve_documents

In notebooks (run from ``examples/tutorials/``)::

    from tutorial_examples.adaptive_rag import adaptive_rag_actions, retrieve_documents
"""
